const std = @import("std");
const tensorError = @import("error.zig").tensorError;
const TensorError = @import("error.zig").TensorError;

fn numItems(shape: []usize) usize {
    var total: usize = 1;
    for (shape) |item| {
        total *= item;
    }
    return total;
}

const PaddingLength = struct { len_raw: usize, len_padded: usize };
pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T,
        allocator: std.mem.Allocator,
        shape: []usize,
        stride: []usize,
        len: usize,

        const Self = @This();

        // TODO: add back "writer" instead of print
        // TODO: loop back to this once there's a method 'index' for partial indices, will be cleaner.
        pub fn printDim(self: *Self, dim: usize, offset: usize) !void {
            if (dim == self.shape.len - 1) {
                std.debug.print("[", .{});
                for (0..self.shape[dim]) |i| {
                    std.debug.print("{d}", .{self.data[offset + i]});
                    if (i < self.shape[dim] - 1) std.debug.print(", ", .{});
                }
                std.debug.print("]", .{});
            } else {
                std.debug.print("[", .{});
                for (0..self.shape[dim]) |i| {
                    try self.printDim(dim + 1, offset + i * self.stride[dim]);
                    if (i < self.shape[dim] - 1) std.debug.print(",\n", .{});
                }
                std.debug.print("]", .{});
            }
        }

        pub fn format(self: Self, writer: *std.Io.Writer) !void {
            try writer.writeAll("[");

            for (self.data) |_| {}
            try writer.print("tensor()", .{});
        }

        pub fn ensureValidShape(data: []T, total: usize) !void {
            if (data.len != total) {
                return tensorError(TensorError.SHAPE_MISMATCH);
            }
        }

        pub fn getCommonDim(self: *Self, tens: *Tensor(T)) !usize {
            const common_dim = self.shape[self.shape.len - 1];
            if (common_dim != tens.shape[0]) {
                return tensorError(TensorError.SHAPE_MISMATCH);
            }
            return common_dim;
        }
        pub fn getVecWidth() comptime_int {
            return 128 / @bitSizeOf(T);
        }

        pub fn getPaddingLength(shape: []usize) PaddingLength {
            // get the padded length of the tensor
            const len_raw = numItems(shape);
            const vec_width = comptime getVecWidth();
            const len_padded = (len_raw + vec_width - 1) / vec_width * vec_width;
            return PaddingLength{ .len_raw = len_raw, .len_padded = len_padded };
        }

        pub fn empty(shape: []usize, allocator: std.mem.Allocator) !Self {
            const vec_width = getVecWidth();
            const N = shape[shape.len - 1];
            const N_padded = (N + vec_width - 1) / vec_width * vec_width;
            const num_rows = numItems(shape) / N;
            const total_padded = num_rows * N_padded;

            const data = try allocator.alloc(T, total_padded);
            const stride = try getStride(shape, allocator);
            const shape_copy = try allocator.dupe(usize, shape);
            return Self{ .data = data, .shape = shape_copy, .allocator = allocator, .stride = stride, .len = numItems(shape) };
        }

        pub fn zeros(shape: []usize, allocator: std.mem.Allocator) !Self {
            const self = try Self.empty(shape, allocator);
            @memset(self.data, 0);
            return self;
        }

        pub fn rand(shape: []usize, allocator: std.mem.Allocator) !Self {
            const self = try Self.empty(shape, allocator);
            var prng = std.Random.DefaultPrng.init(42);
            const random = prng.random();
            for (self.data) |*item| {
                item.* = random.float(T);
            }
            return self;
        }
        pub fn fromSlice(data: []T, shape: []usize, allocator: std.mem.Allocator) !Self {
            const padding_length = getPaddingLength(shape);
            try ensureValidShape(data, padding_length.len_raw);

            const vec_width = getVecWidth();
            const N = shape[shape.len - 1];
            const N_padded = (N + vec_width - 1) / vec_width * vec_width;
            const num_rows = padding_length.len_raw / N;

            const padded_data = try allocator.alloc(T, num_rows * N_padded);
            @memset(padded_data, 0);

            for (0..num_rows) |row| {
                const src = data[row * N .. row * N + N];
                const dst = padded_data[row * N_padded .. row * N_padded + N];
                @memcpy(dst, src);
            }

            const stride = try getStride(shape, allocator);
            const shape_copy = try allocator.dupe(usize, shape);
            return Self{ .data = padded_data, .shape = shape_copy, .allocator = allocator, .stride = stride, .len = padding_length.len_raw };
        }
        fn getStride(shape: []usize, allocator: std.mem.Allocator) ![]usize {
            const stride = try allocator.alloc(usize, shape.len);

            // init stride is always 1
            stride[shape.len - 1] = 1;

            var i: usize = shape.len - 1;
            while (i > 0) {
                i -= 1;
                stride[i] = stride[i + 1] * shape[i + 1];
            }

            return stride;
        }

        fn getFlatIndex(self: *Self, indices: []const usize) usize {
            var flat_index: usize = 0;
            for (indices, 0..) |idx, i| {
                flat_index += idx * self.getDataStride(i);
            }
            return flat_index;
        }

        fn getDataStride(self: *Self, dim: usize) usize {
            const vec_width = getVecWidth();
            var result: usize = 1;
            var i: usize = self.shape.len - 1;
            while (i > dim) : (i -= 1) {
                const padded = (self.shape[i] + vec_width - 1) / vec_width * vec_width;
                result *= padded;
            }
            return result;
        }

        pub fn at(self: *Self, indices: []const usize) T {
            // example:
            // shape 3, 4, 5
            // [[[6, 2, 9, 6, 8],
            //   [2, 3, 5, 3, 2],
            //   [7, 3, 1, 4, 2],
            //   [0, 8, 3, 3, 1]],
            //
            //  [[8, 3, 5, 3, 7],
            //   [0, 4, 8, 2, 5],
            //   [9, 8, 2, 7, 0],
            //   [8, 9, 3, 2, 2]],
            //
            //  [[3, 4, 4, 5, 7],
            //   [2, 6, 1, 1, 0],
            //   [3, 5, 7, 1, 2],
            //   [4, 3, 8, 0, 7]]])
            //   stride(20, 5, 1)
            //   flat_index = i * 20 + j * 5 + k * 1

            const flat_index = self.getFlatIndex(indices);
            const data = self.data[flat_index];
            return data;
        }

        pub fn set(self: *Self, indices: []const usize, value: T) void {
            const flat_index = self.getFlatIndex(indices);
            self.data[flat_index] = value;
        }
        pub fn matmul(self: *Self, tens: *Tensor(T), allocator: std.mem.Allocator) !Tensor(T) {

            // go through rows 0..end(mat1)
            //      each row, get each item in mat1, each item at mat2.at(row_num, i)
            //
            //                shape(2, 3)
            // [[1,2,3],      <-- [1,2,3,4,5,6]
            // [4,5,6]]       stride(3,1)
            //                [[1,2,3,][4,5,6]]
            // @
            //                shape(3, 2)
            // [[1, 2],       <-- [1,2,3,4,5,6]
            // [3, 4],        stride(2,1)
            // [5, 6]]        [[1,2,][3,4,][5,6]]
            // a[-1] has to equal b[2];

            // var new_shape = [_]usize{};
            // var result = try Tensor(T).empty(.{});
            const M = self.shape[0]; // 2
            const K = self.shape[1]; // 3
            const N = tens.shape[1]; // 2
            // std.debug.print("M val: {any}", .{M});
            // std.debug.print("N val: {any}", .{N});
            // std.debug.print("K val: {any}", .{K});
            if (K != tens.shape[0]) return tensorError(TensorError.SHAPE_MISMATCH); // 3
            var result_shape = [_]usize{ M, N };
            var result = try Tensor(T).zeros(&result_shape, allocator);
            // std.debug.print("result shape: {any}\n", .{result.shape});
            // std.debug.print("result data len: {}\n", .{result.data.len});
            // std.debug.print("result getDataStride(0): {}\n", .{result.getDataStride(0)});
            // slow path, when indexing the tens tensor we're moving like it's col major
            // on cpu, a 64 byte l1 cache line gets loaded for each lookup. in larger tensors
            // this version will waste those free lookups. zig compiler is pretty smart though
            // it does a lot of optimization even when we don't
            // for (0..M) |i| { // 0, 1
            //     for (0..N) |j| { // 0, 1
            //         var total: T = 0;
            //         for (0..K) |k| { // 0..3
            //             // [i, j, k]
            //             // [0, 0, 0]
            //             // [0, 0, 1]
            //             // [0, 0, 2]
            //             // [0, 1, 0]
            //             // ...
            //
            //             total += self.at(&.{ i, k }) * tens.at(&.{ k, j });
            //             // self.data = [1,2,3,4,5,6] -------- tens.data = [1,2,3,4,5,6]
            //             // shaped self.data = [[1,2,3],  shaped tens.data = [[1,2],
            //             //                     [4,5,6]]                      [3,4],
            //             //                                                   [5,6]]
            //             //
            //
            //         }
            //         result.set(&.{ i, j }, total);
            //     }
            // }
            //
            // for (0..M) |i| { // 0, 1
            //     for (0..K) |k| { // 0, 1
            //         const a = self.at(&.{ i, k });
            //         // broadcast to a vector of vec_width copies
            //         const vec_width = comptime getVecWidth();
            //         const bcast_vec: @Vector(vec_width, T) = @splat(a);
            //         const t_offset = k * tens.stride[0];
            //         const r_offset = i * result.stride[0];
            //         var j: usize = 0;
            //
            //         const N_padded = (N + vec_width - 1) / vec_width * vec_width;
            //         while (j < N_padded) : (j += vec_width) {
            //             const t_vec: @Vector(vec_width, T) = tens.data[t_offset + j ..][0..vec_width].*;
            //             var r_vec: @Vector(vec_width, T) = result.data[r_offset + j ..][0..vec_width].*;
            //             r_vec = r_vec + bcast_vec * t_vec;
            //
            //             // a lot of little writes here
            //             // next item to fix
            //             // probably need to swap back ikj to ijk now
            //             // that it's actually vectorized
            //             result.data[r_offset + j ..][0..vec_width].* = r_vec;
            //         }
            //     }
            // }
            //
            // const padding_len_tens = getPaddingLength(tens.shape);
            // const padding_len_self = getPaddingLength(self.shape);
            const vec_width = getVecWidth();
            const N_padded = tens.getDataStride(0);
            // std.debug.print("tens padding: {any}\n", .{padding_len_tens});
            // std.debug.print("self padding: {any}\n", .{padding_len_self});
            // std.debug.print("vec_width: {any}\n", .{vec_width});
            // std.debug.print("n_padded: {any}\n", .{N_padded});

            for (0..M) |i| {
                const r_offset = result.getDataStride(0) * i;
                var j: usize = 0;
                while (j < N_padded) : (j += vec_width) {
                    var r_vec: @Vector(vec_width, T) = @splat(0);
                    for (0..K) |k| {
                        const a = self.at(&.{ i, k });
                        // std.debug.print("a: {any}\n", .{a});
                        const bcast_vec: @Vector(vec_width, T) = @splat(a);
                        // std.debug.print("bcast_vec: {any}\n", .{bcast_vec});
                        const t_offset = k * tens.getDataStride(0);
                        // std.debug.print("t_offset: {any}\n", .{t_offset});
                        const t_vec: @Vector(vec_width, T) = tens.data[t_offset + j ..][0..vec_width].*;
                        // std.debug.print("t_vec: {any}\n", .{t_vec});
                        r_vec += bcast_vec * t_vec;
                        // std.debug.print("r_vec: {any}\n", .{r_vec});

                        // self.data(raw)    = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 } len: 35
                        // self.data(padded) = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0} len: 36
                        //
                        // tens.data(raw)    = { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 } len: 15
                        // tens.data(padded) = { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0 } len: 16
                        // shaped self.data = [[1,2,3,4,5],  shaped tens.data =  [[1,2,3],   [M=7,K=5] @ [K=5,N=3]
                        //                     [1,2,3,4,5],                       [1,2,3],
                        //                     [1,2,3,4,5],                       [1,2,3],
                        //                     [1,2,3,4,5],                       [1,2,3],
                        //                     [1,2,3,4,5],                       [1,2,3]]
                        //                     [1,2,3,4,5],
                        //                     [1,2,3,4,5]]
                        //              res = [[15, 30, 45], padded tens.data = [[1,2,3,0], [M=7,K=5] @ [K=5, N=4]
                        //                     [15, 30, 45],                     [1,2,3,0],
                        //                     [15, 30, 45],                     [1,2,3,0],
                        //                     [15, 30, 45],                     [1,2,3,0],
                        //                     [15, 30, 45],                     [1,2,3,0],
                        //                     [15, 30, 45],                     [1,2,3,0]]
                        //                     [15, 30, 45]]
                        //
                        //                     [(1+2+3+4+5), (2+4+6+8+10), (3+6+9+12+15), (0+0+0+0+0)]
                        //
                        //                     (i=0,k=2)        splat = {3, 3, 3, 3};
                        //                     (2(k)*4(vec_w))  t_offset = 8
                        //
                    }
                    result.data[r_offset + j ..][0..vec_width].* = r_vec;
                }
            }

            return result;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.stride);
        }
    };
}

test "tensor empty has correct shape" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 3), tens.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), tens.shape[1]);
    try std.testing.expectEqual(@as(usize, 12), tens.len);
}

test "tensor zeros are all zero" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var tens = try Tensor(f32).zeros(&shape, gpa);
    defer tens.deinit();

    for (tens.data) |item| {
        try std.testing.expectEqual(@as(f32, 0), item);
    }
}

test "tensor at returns correct value" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var tens = try Tensor(f32).fromSlice(&data, &shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(f32, 1), tens.at(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2), tens.at(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 5), tens.at(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 12), tens.at(&.{ 2, 3 }));
}

test "tensor sets correct value" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var tens = try Tensor(f32).fromSlice(&data, &shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(f32, 6), tens.data[5]);

    tens.set(&.{ 1, 1 }, 900);

    try std.testing.expectEqual(@as(f32, 900), tens.data[5]);
}

test "tensor fromSlice has correct data" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var data = [_]f32{ 1, 2, 3, 4 };
    var tens = try Tensor(f32).fromSlice(&data, &shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 4), tens.len);
    try std.testing.expectEqual(@as(f32, 1), tens.data[0]);
    try std.testing.expectEqual(@as(f32, 2), tens.data[1]);
    try std.testing.expectEqual(@as(f32, 3), tens.data[4]);
    try std.testing.expectEqual(@as(f32, 4), tens.data[5]);
}

test "tensor dtype f16 zeros" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var tens = try Tensor(f16).zeros(&shape, gpa);
    defer tens.deinit();

    for (tens.data) |item| {
        try std.testing.expectEqual(@as(f16, 0), item);
    }
}

test "tensor stride are correct for 2d" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 4), tens.stride[0]);
    try std.testing.expectEqual(@as(usize, 1), tens.stride[1]);
}

test "tensor stride are correct for 3d" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4, 5 };
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 20), tens.stride[0]);
    try std.testing.expectEqual(@as(usize, 5), tens.stride[1]);
    try std.testing.expectEqual(@as(usize, 1), tens.stride[2]);
}

test "tensor stride are correct for 1d" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{6};
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 1), tens.stride[0]);
}

test "matmul produces correct value" {
    const gpa = std.testing.allocator;
    var mat_1_shape = [_]usize{ 3, 3 };
    var mat_2_shape = [_]usize{ 3, 3 };
    var data_arr_1 = [_]f32{ 1, 2, 3, 1, 2, 3, 1, 2, 3 };
    var data_arr_2 = [_]f32{ 1, 2, 3, 1, 2, 3, 1, 2, 3 };
    var F32FromSlice1: Tensor(f32) = try Tensor(f32).fromSlice(&data_arr_1, &mat_1_shape, gpa);
    var F32FromSlice2: Tensor(f32) = try Tensor(f32).fromSlice(&data_arr_2, &mat_2_shape, gpa);
    defer F32FromSlice1.deinit();
    defer F32FromSlice2.deinit();
    var res = try F32FromSlice1.matmul(&F32FromSlice2, gpa);
    defer res.deinit();

    try std.testing.expectEqualSlices(f32, &[_]f32{ 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0 }, res.data);
}

const std = @import("std");
const graditude = @import("graditude");
const Tensor = @import("tensor.zig").Tensor;

// pub fn matmul(x: Tensor, y: Tensor) Tensor {
//     unreachable;
// }

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var shape_arr = [_]usize{
        3,
        4,
    };
    var shape_arr_T = [_]usize{ 4, 3 };
    var F32Tens = try Tensor(f32).empty(&shape_arr, gpa);
    defer F32Tens.deinit();

    _ = &F32Tens;
    for (F32Tens.data) |item| {
        _ = item * 4;
    }

    var F16Zeros = try Tensor(f16).zeros(&shape_arr, gpa);
    defer F16Zeros.deinit();

    const val = F16Zeros.at(&.{ 0, 0 });

    std.debug.print("zeros data: {any}\n", .{F16Zeros.data});
    std.debug.print("break\n", .{});
    std.debug.print("val: {any}\n", .{val});

    var slice_arr = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    var F32FromSlice = try Tensor(f32).fromSlice(&slice_arr, &shape_arr, gpa);
    defer F32FromSlice.deinit();

    var F32FromSlice2 = try Tensor(f32).fromSlice(&slice_arr, &shape_arr_T, gpa);
    defer F32FromSlice2.deinit();

    std.debug.print("fromslice data: {any}\n", .{F32FromSlice.data});
    std.debug.print("break\n", .{});
    std.debug.print("stride data: {any}\n", .{F32FromSlice.stride});
    std.debug.print("break\n", .{});
    std.debug.print("fmt: {f}", .{F32FromSlice});
    std.debug.print("break\n", .{});
    try F32Tens.printDim(1, 0);
    const at_result = F32FromSlice.at(&.{1});
    std.debug.print("stride data: {any}\n", .{at_result});
    std.debug.print("break\n", .{});
    F32FromSlice.set(&.{ 1, 1 }, 900);
    std.debug.print("fromslice data: {any}\n", .{F32FromSlice.data});
    std.debug.print("break\n", .{});
    const idx1 = F32FromSlice.at(&.{ 1, 0 });
    std.debug.print("idx: {}", .{idx1});
    std.debug.print("break\n", .{});
    var data_arr = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var data_arr2 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var mat_1_shape = [_]usize{ 2, 3 };
    var mat_2_shape = [_]usize{ 3, 2 };
    var F32FromSlice3: Tensor(f32) = try Tensor(f32).fromSlice(&data_arr, &mat_1_shape, gpa);
    var F32FromSlice4: Tensor(f32) = try Tensor(f32).fromSlice(&data_arr2, &mat_2_shape, gpa);
    defer F32FromSlice3.deinit();
    defer F32FromSlice4.deinit();

    const f32_type = @TypeOf(F32FromSlice4);
    std.debug.print("idx: {}\n\n", .{f32_type});
    std.debug.print("before1: {any}\n\n", .{F32FromSlice3.data});
    std.debug.print("before2: {any}\n\n", .{F32FromSlice4.data});
    const start = std.Io.Clock.real.now(io);
    var res = try F32FromSlice3.matmul(&F32FromSlice4, gpa);
    const finish = std.Io.Clock.real.now(io);
    const elapsed: f64 = (@as(f64, @floatFromInt(finish.nanoseconds)) - @as(f64, @floatFromInt(start.nanoseconds))) / 1_000_000.00;

    std.debug.print("elapsed ms : {any}\n\n", .{elapsed});
    defer res.deinit();
}

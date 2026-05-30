const std = @import("std");

pub const TensorError = error{SHAPE_MISMATCH};

pub fn tensorError(err: TensorError) TensorError {
    return err;
}

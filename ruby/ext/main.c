#include "ruby.h"

int add(int a, int b) {
    return a + b;
}

static VALUE rb_add(VALUE self, VALUE rb_a, VALUE rb_b) {
    int a = NUM2INT(rb_a);
    int b = NUM2INT(rb_b);
    int result = add(a, b);
    return INT2NUM(result);
}

void Init_theogradz(void) {
    VALUE mod = rb_define_module("MyMath");
    rb_define_singleton_method(mod, "add", rb_add, 2);
}

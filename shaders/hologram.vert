#version 330

in vec3 in_position;
uniform mat4 Mvp;

void main() {
    gl_Position = Mvp * vec4(in_position, 1.0);
}

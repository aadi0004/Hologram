#version 330

uniform mat4 Mvp;
uniform mat4 Model;
uniform float Time;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal_ws;
out vec3 v_pos_ws;

void main() {
    vec4 pos_ws = Model * vec4(in_position, 1.0);
    v_pos_ws = pos_ws.xyz;
    v_normal_ws = mat3(Model) * in_normal;

    gl_Position = Mvp * vec4(in_position, 1.0);
}

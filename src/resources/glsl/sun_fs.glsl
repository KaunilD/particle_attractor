#version 440 core

uniform highp vec3 color;

out vec4 gl_FragColor;

void main()
{
    gl_FragColor = vec4(color, 1.0f);
}



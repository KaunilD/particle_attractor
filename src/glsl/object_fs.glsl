#version 440 core

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct Material {
    float shininess;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Material material;
uniform Light light;

uniform sampler2D textureSampler;

uniform highp mat3 normalMat;
uniform highp vec3 cameraEye;

in vec3 vs_normal;
in vec4 vs_position;
in vec4 vs_velocity;

out vec4 gl_FragColor;

void main()
{

    vec3 ambient = texture(textureSampler, vs_position.xy + vec2(0.5)).xyz * material.ambient;

    // diffuse 
    vec3 norm = normalize(vs_normal);
    vec3 lightDir = normalize(light.position - vs_position.xyz);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material.diffuse);

    // specular
    vec3 viewDir = normalize(cameraEye - vs_position.xyz);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);

    vec3 result = ambient + diffuse + specular;
    gl_FragColor = vec4(result, 1.0);
}



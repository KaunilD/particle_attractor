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

uniform sampler2D textureSampler;

uniform highp mat3 normalMat;
uniform highp vec3 cameraEye;

uniform Material material;
uniform Light light;

in vec3 vs_normal;
in vec4 vs_position;

out vec4 gl_FragColor;

void main()
{
    /*
        ambient lighting
    */ 
    vec3 ambient = material.ambient * light.ambient;
    /*
        diffuse lighting
    */
    vec3 normal = normalize(vs_normal);
    vec3 lightDirection = normalize(-light.position);
    float diffuseStrength = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = (diffuseStrength * material.diffuse) * light.diffuse;
    /*
        specular lighting
    */
    vec3 viewDirection = normalize(cameraEye - vs_position.xyz);
    vec3 reflectDirection = reflect(-lightDirection, normal);
    
    float specularStrength = pow(max(dot(viewDirection, reflectDirection), 0.0), material.shininess);
    vec3 specular = (specularStrength * material.specular) * light.specular;

    vec3 result = (ambient + diffuse + specular);
    gl_FragColor = vec4(result, 1.0f);
}



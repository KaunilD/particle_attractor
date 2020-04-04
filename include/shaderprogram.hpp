#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include "libs.hpp"
#include "camera.hpp"

class ShaderProgram{
	public:

		ShaderProgram();

		GLuint program;

		void loadShaders(const char * vs_path, const char * fs_path);
		void activate();
		void deactivate();

		
		void ShaderProgram::setMat4(std::string name, glm::mat4 matrix) {
			glUniformMatrix4fv(
				glGetUniformLocation(program, name.c_str()),
				1,
				GL_FALSE,
				glm::value_ptr(matrix)
			);
		}

		void ShaderProgram::setVec3(std::string name, glm::vec3 vector) {
			glUniform3fv(
				glGetUniformLocation(program, name.c_str()),
				1,
				glm::value_ptr(vector)
			);
		}
		void ShaderProgram::setFloat(std::string name, float val) {
			glUniform1f(
				glGetUniformLocation(program, name.c_str()),
				val
			);
		}

		void checkCompileErrors(unsigned int shader, std::string type);

};

#endif // SHADERPROGRAM_H

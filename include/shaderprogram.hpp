#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include "libs.hpp"
#include "camera.hpp"

class ShaderProgram{
	public:

		ShaderProgram();
		ShaderProgram(QObject * parent);

		unique_ptr<QOpenGLShaderProgram> program;

		void loadShaders(const char * vs_path, const char * fs_path);
		bool activate();
		void deactivate();

		
		
		template <typename T>
		void setUniform(const std::string& t_name, const T& t_val) {
			assert(this->program->isLinked(), "ShaderProgram:: Setting variable of an unlinked Program!");
			GLuint id = program->uniformLocation(t_name.c_str());
			program->setUniformValue(id, t_val);
		};

};

#endif // SHADERPROGRAM_H

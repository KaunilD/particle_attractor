#include "shaderprogram.hpp"

ShaderProgram::ShaderProgram(){
	qDebug() << "Shader Created";
}

ShaderProgram::ShaderProgram(QObject *parent){
	qDebug() << "Shader Created";
	program = make_unique<QOpenGLShaderProgram>(parent);

};

void ShaderProgram::loadShaders(const char * vsPath, const char * fsPath) {
	program->addShaderFromSourceFile(QOpenGLShader::Vertex, QString(vsPath));
	program->addShaderFromSourceFile(QOpenGLShader::Fragment, QString(fsPath));
}

void ShaderProgram::deactivate() {
	if (this->program->isLinked()) {
		this->program->release();
	}
}

bool ShaderProgram::activate() {
	this->program->link();
	if (this->program->isLinked()) {
		return this->program->bind();
	}
	return false;
}


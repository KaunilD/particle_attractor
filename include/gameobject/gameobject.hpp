#ifndef GAMEOBJECT_H
#define GAMEOBJECT_H

#include "libs.hpp"
#include "vertex.hpp"
#include "shaderprogram.hpp"
#include "algorithms/algorithms.hpp"

class GameObject: protected QOpenGLFunctions {

	public:
	GameObject();
	GameObject(
		bool npc,
		float mass,
		QVector3D position,
		QVector3D color
	);
	virtual ~GameObject();

	QMatrix4x4 & getModelMatrix();
	QMatrix4x4 & getProjectionMatrix();

	void setupModelMatrix(QVector3D translate, QVector3D scale);
	void setScale(QVector3D scale);
	void setTranslate(QVector3D translate);
	void setPosition(QVector3D translate);


	void setColor(QVector3D color);
	QVector3D getColor() { return color; };

	QVector3D getPosition() { return position; };

	void setupGLBuffers();
	void render(ShaderProgram * shaderProgram);
	void loadObject(QString objFile, QString textureImage);

	void updateObject(int frames, QKeyEvent * event, Algorithms * algorithm);

	QVector3D velocity = QVector3D(0.0, 0.0, 0.0);
	QVector3D acceleration = QVector3D(0, 0, 0);
	float mass, g = 0.4f, angle = 0, angularVelocity = 0, angularAcceleration = 0.01;
	QVector3D * sqrt3D(QVector3D *);
	void updatePosition(QVector3D);
	void setVelocity(QVector3D  const &);
	void setAcceleration(QVector3D const &);
	void setMass(float);

	float getForceVector(GameObject * p);
	void applyForceVector(float, QVector3D);
	bool npc;

	QMatrix4x4 * modelMatrix, *projectionMatrix;
	QVector3D position, color, scale;
	QOpenGLTexture * texture = NULL;

	QVector<Vertex> vertices;
	QVector<GLsizei> indices;

	QVector<QVector3D> rawVertices;
	QVector<QVector2D> rawTextures;
	QVector<QVector3D> rawNormals;

	std::vector<int> vertesIndices;
	std::vector<int> textureIndices;
	std::vector<int> normalIndices;

	shared_ptr<QOpenGLBuffer> attributeBuffer, indexBuffer;


};
#endif
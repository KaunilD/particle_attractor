#ifndef GAMEOBJECT_H
#define GAMEOBJECT_H

#include "libs.hpp"
#include "vertex.hpp"
#include "gameobject/mesh.hpp"
#include "gameobject/material.hpp"
#include "shaderprogram.hpp"
#include "algorithms/algorithms.hpp"

class GameObject: protected QOpenGLFunctions {

	public:
	GameObject();
	GameObject(bool, float, Mesh, Material);
	~GameObject() = default;
	QMatrix4x4* modelMatrix;
	QVector3D position, color, scale;

	Mesh m_mesh;
	Material m_material;

	QMatrix4x4 & getModelMatrix() const;
	QMatrix4x4 & getProjectionMatrix();

	void setupModelMatrix(QVector3D translate, QVector3D scale);
	
	void setScale(QVector3D scale);
	void setTranslate(QVector3D translate);
	void setPosition(QVector3D translate);


	void setColor(QVector3D color);
	QVector3D getColor() { return color; };

	QVector3D getPosition() { return position; };

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
};
#endif
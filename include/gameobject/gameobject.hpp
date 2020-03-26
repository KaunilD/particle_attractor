#ifndef GAMEOBJECT_H
#define GAMEOBJECT_H

#include "libs.hpp"
#include "vertex.hpp"
#include "shaderprogram.hpp"
#include "gameobject/mesh.hpp"
#include "gameobject/material.hpp"

class GameObject: protected QOpenGLFunctions {

	public:
	GameObject();
	GameObject(bool);
	~GameObject();

	bool m_npc;
	QVector3D velocity = QVector3D(1.0, 2.0, -1.0);
	QVector3D acceleration = QVector3D(0, 0, 0);
	float m_mass, g = 0.4f, angle = 0, angularVelocity = 0, angularAcceleration = 0.01;


	unique_ptr<QMatrix4x4> m_modelMatrix;
	QVector3D m_position, m_color, m_scale;

	shared_ptr<Mesh> m_mesh;
	shared_ptr<Material> m_material;

	QMatrix4x4& getModelMatrix() const { return *m_modelMatrix.get(); };

	void setupModelMatrix(QVector3D translate, QVector3D scale);
	void updateModelMatrix();

	void setMass(float t_mass);
	void setMesh(shared_ptr<Mesh> t_mesh);
	void setMaterial(shared_ptr<Material> t_material);
	void setScale(QVector3D scale);
	void setPosition(QVector3D translate);


	void setColor(QVector3D color);
	QVector3D getColor() { return m_color; };

	QVector3D getPosition() const { return m_position; };

	virtual void updateObject(int frames, QKeyEvent * event, const GameObject& referenceObj) = 0;

	void updatePosition(QVector3D);
	void setVelocity(QVector3D  const &);
	void setAcceleration(QVector3D const &);

	float getForceVector(const GameObject& p) const;
	void applyForceVector(float, QVector3D);
};
#endif
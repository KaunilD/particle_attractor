#include "gameobject/gameobject.hpp"

GameObject::GameObject() {}

GameObject::GameObject(
	bool _npc, float _mass, QVector3D _position, QVector3D _color): npc(_npc), mass(_mass), position(_position), color(_color){
	initializeOpenGLFunctions();
	float scale = qBound(7.f, mass*10, 10.0f);
	setupModelMatrix(
		position,
		QVector3D(scale, scale, scale)
	);
}

void GameObject::loadObject(QString filePath, QString textureImage) {
	qDebug() << "ObjLoader:: Reading object" << filePath;

	QFile obj_file(filePath);
	obj_file.open(QIODevice::ReadOnly);

	QTextStream textStream(&obj_file);


	if (obj_file.isOpen()) {
		while (!textStream.atEnd()) {
			QString line = textStream.readLine();
			QStringList list = line.split(" ");
			bool isTexture = list.count() == 4 ? false : true;
			if (list[0] == "v" && !isTexture) {
				QVector3D vertex;
				for (int i = 1; i < list.count(); i++) {
					vertex[i - 1] = list[i].toFloat();
				}
				//qDebug() << vertex;
				rawVertices.push_back(vertex);
			}
			else if (list[0] == "vt" && isTexture) {
				QVector2D texture;
				for (int i = 1; i < list.count(); i++) {
					texture[i - 1] = list[i].toFloat();
				}
				rawTextures.push_back(texture);
			}
			else if (list[0] == "vn" && !isTexture) {
				QVector3D normal;
				for (int i = 1; i < list.count(); i++) {
					normal[i - 1] = list[i].toFloat();
				}
				//qDebug() << normal;
				rawNormals.push_back(normal);
			}
			else if (list[0] == "f" && !isTexture) {
				// f 1/1/1 2/2/2 3/3/3
				for (int i = 1; i < list.count(); i++) {
					QStringList indexGroup = list[i].split("/");
					vertesIndices.push_back(indexGroup[0].toInt());
					textureIndices.push_back(indexGroup[1].toInt());
					normalIndices.push_back(indexGroup[2].toInt());
				}
				//qDebug() << list;
			}

		}
	}

	for (int i = 0; i < vertesIndices.size(); i++) {
		vertices.push_back(Vertex{
				rawVertices[vertesIndices[i] - 1],
				rawNormals[normalIndices[i] - 1],
				rawTextures[textureIndices[i] - 1],
				color,
				QVector3D(1.0f, 1.0f, 1.0f),
				QVector3D(1.0f, 1.0f, 1.0f),
			});
		indices.push_back(i);
	}
	
	if(textureImage!=NULL)
		texture = new QOpenGLTexture(QImage(textureImage).mirrored());
}


void GameObject::setColor(QVector3D _color) {
	color = _color;
}

void GameObject::setPosition(QVector3D _position) {
	position = _position;
}

void GameObject::setTranslate(QVector3D translate) {
	modelMatrix->translate(translate);
}

void GameObject::setScale(QVector3D _scale) {
	scale = _scale;
	modelMatrix->scale(_scale);
}

void GameObject::setupModelMatrix(QVector3D _translate, QVector3D _scale) {
	modelMatrix = new QMatrix4x4();
	modelMatrix->setToIdentity();

	setScale(_scale);
	setTranslate(_translate);
}
void GameObject::setupGLBuffers() {
	attributeBuffer = make_shared<QOpenGLBuffer>();
	attributeBuffer->create();
	attributeBuffer->bind();
	attributeBuffer->allocate(
		vertices.constData(),
		vertices.count()*static_cast<int>(sizeof(Vertex))
	);

	indexBuffer = make_shared<QOpenGLBuffer>(QOpenGLBuffer::IndexBuffer);
	indexBuffer->create();
	indexBuffer->bind();
	indexBuffer->allocate(
		indices.constData(),
		indices.count() * static_cast<int>(sizeof(GL_UNSIGNED_INT))
	);

}

void GameObject::render(ShaderProgram * shaderProgram) {
	GLuint textureAttribLoc, positionAttribLoc, normalAttribLoc, colorAttribLoc;
	attributeBuffer->bind();
	indexBuffer->bind();
	
	// POSITION
	positionAttribLoc = shaderProgram->program->attributeLocation("vertex_position");
	shaderProgram->program->enableAttributeArray(positionAttribLoc);
	shaderProgram->program->setAttributeBuffer(positionAttribLoc, GL_FLOAT, (int)offsetof(Vertex, position), 3, sizeof(Vertex));

	// NORMAL
	normalAttribLoc = shaderProgram->program->attributeLocation("vertex_normal");
	shaderProgram->program->enableAttributeArray(normalAttribLoc);
	shaderProgram->program->setAttributeBuffer(normalAttribLoc, GL_FLOAT, (int)offsetof(Vertex, normals), 3, sizeof(Vertex));

	// COLOR
	colorAttribLoc = shaderProgram->program->attributeLocation("vertex_color");
	shaderProgram->program->enableAttributeArray(colorAttribLoc);
	shaderProgram->program->setAttributeBuffer(colorAttribLoc, GL_FLOAT, (int)offsetof(Vertex, color), 3, sizeof(Vertex));

	if (texture!=NULL) {

		texture->bind();
		// TEXTURE
		textureAttribLoc = shaderProgram->program->attributeLocation("vertex_texture");
		shaderProgram->program->enableAttributeArray(textureAttribLoc);
		shaderProgram->program->setAttributeBuffer(textureAttribLoc, GL_FLOAT, (int)offsetof(Vertex, texture), 2, sizeof(Vertex));
	}


	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(positionAttribLoc);
	glDisableVertexAttribArray(normalAttribLoc);
	glDisableVertexAttribArray(colorAttribLoc);
	if (texture != NULL) {
		glDisableVertexAttribArray(textureAttribLoc);

	}
}

QMatrix4x4 & GameObject::getModelMatrix() {
	return *modelMatrix;
}



GameObject::~GameObject() {

	indexBuffer->destroy();
	attributeBuffer->destroy();

}


void GameObject::updateObject(
	int frames, QKeyEvent * event, Algorithms * attractor
) {

}


float GameObject::getForceVector(GameObject * p) {
	QVector3D distance = p->getPosition() - position;
	float magnitude = sqrt(pow(distance.x(), 2) + pow(distance.y(), 2) + pow(distance.z(), 2));
	magnitude = qBound(5.f, magnitude, 20.f);
	return ( g * mass * p->mass) / (magnitude * magnitude);
}

QVector3D * GameObject::sqrt3D(QVector3D *a) {
	return new QVector3D(
		sqrt(a->x()), sqrt(a->y()), sqrt(a->z())
	);
}


void GameObject::applyForceVector(float force, QVector3D distance) {
	QVector3D resForce = (distance.normalized())*force;
	acceleration += (resForce / mass);
	velocity += acceleration*(1/60.f);
	position += velocity * (1 / 60.f);
	acceleration *= 0;
	setupModelMatrix(position, scale);
}

void GameObject::setVelocity(QVector3D const &_velocity) {
	velocity = _velocity;
}
void GameObject::setAcceleration(QVector3D const &_acceleration) {
	acceleration = _acceleration;
}

void GameObject::setMass(float _mass) {
	mass = _mass;
}
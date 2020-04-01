#include "oglwidget.hpp"

OGLWidget::OGLWidget(QWidget *parent) : QOpenGLWidget(parent)
{
	
	QSurfaceFormat format;
	format.setSamples(16);
	format.setProfile(QSurfaceFormat::CompatibilityProfile);
	this->setFormat(format);
	this->setFocus();

	updateGLTimer.setInterval(1000 / 60.f);
	connect(&updateGLTimer, SIGNAL(timeout()), this, SLOT(update()));

	frameTimeUpdateTimer.setInterval(1000);
	connect(&frameTimeUpdateTimer, SIGNAL(timeout()), this, SLOT(frameTimeUpdateTimerTicked()));
	
}


OGLWidget::~OGLWidget() {
	LOG("OGLWidget::Destroyed");
}

void OGLWidget::initializeGL() {

	qDebug() << "initializeOGL";
	initializeOpenGLFunctions();

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	
	camera = make_shared<Camera>(
		QVector3D(0.0f, 0.0f, 10.0f),
		QVector3D(0.0f, 0.0f, -1.0f),
		QVector3D(0.0f, 1.0f, 0.0f),
		90.0f, width(), height(), 0.001f, 100.f
	);
	camera->setSpeed(0.5);

	particleObjShader = make_shared<ShaderProgram>(this);
	particleObjShader->loadShaders(
		":/vertexShader",
		":/fragShader"
	);
	
	particleRenderer = make_shared<ParticleRenderer>();
	
	initializeGLfromGrid();
	
	elapsedTimer.start();
	updateGLTimer.start();
	frameTimeUpdateTimer.start();
}

void OGLWidget::frameTimeUpdateTimerTicked() {
	QString perfUpdate("");
	perfUpdate += "FPS: " + QString::number(frame) + " Frame Time: " + QString::number(elapsedTime * 0.000001);
	
	emit fps(perfUpdate);
	
	frame = 0;
	elapsedTimer.restart();
}


void OGLWidget::initializeGLfromGrid() {
	
	gameObjects = make_shared<std::vector<shared_ptr<GameObject>>>();

	m_sphereMesh = make_shared<Mesh>(Utils::readObj(QString(":/sphere")));
	m_sphereMaterial = make_shared<Material>(QString(":/blattTexture"));

	// setup sun
	shared_ptr<SunObject> sunObject = make_shared<SunObject>(true);
	sunObject->setMass(1000.0f);
	sunObject->setMesh(m_sphereMesh);
	sunObject->setMaterial(m_sphereMaterial);
	sunObject->setScale(QVector3D(10.0f, 10.0f, 10.0f));
	gameObjects->push_back(std::move(sunObject));

	// setup particles
	
	shared_ptr<ParticleObject> particleObject;
	for (int i = 1; i < 100; i++) {
		particleObject = make_shared<ParticleObject>(false);
		particleObject->setMass(
			Algorithms::randomInt(10) / 10.f + 0.01
		);
		particleObject->setMesh(m_sphereMesh);
		particleObject->setMaterial(m_sphereMaterial);
		particleObject->setScale(
			QVector3D(
				particleObject->m_mass, particleObject->m_mass, particleObject->m_mass
			)
		);

		particleObject->setPosition(
			QVector3D(
				Algorithms::randomInt(10) - 5,
				Algorithms::randomInt(10) - 5,
				Algorithms::randomInt(10) - 5
			)
		);

		gameObjects->push_back(std::move(particleObject));
	}
	
}


void OGLWidget::paintGL() {

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/*
		update and render gameObjects
	*/
	#pragma omp parallel for
	for (int i = 0; i < gameObjects->size(); i++) {
		gameObjects->at(i)->updateObject(
			frame, 0, *gameObjects->at(0)
		);
	}
	for (int i = 0; i < gameObjects->size(); i++) {

		particleRenderer->render(
			particleObjShader,
			gameObjects->at(i),
			camera
		);
	}
	frame += 1;
	elapsedTime = elapsedTimer.nsecsElapsed();
}

void OGLWidget::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
	camera->updateProjectionMatrix(width, height);
}

void OGLWidget::reset(unsigned int w, unsigned int h) {
	qDebug() << "reset clicked" << w << " " << h;

	updateGLTimer.stop();
	
	initializeGLfromGrid();
	this->setFocus();
	updateGLTimer.start();
}

void OGLWidget::signalGameOver() {
	emit gameOver();
}

void OGLWidget::keyPressEvent(QKeyEvent * event) {
	/*
	for (int i = 0; i < gameObjects->size(); i++) {
		gameObjects->at(i)->updateObject(frame, event);
	}
	*/
}

void OGLWidget::wheelEvent(QWheelEvent * event) {
	camera->update(event);
}
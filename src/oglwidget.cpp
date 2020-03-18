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

	
	camera = new Camera(
		QVector3D(0.0f, 0.0f, 10.0f),
		QVector3D(0.0f, 0.0f, -1.0f),
		QVector3D(0.0f, 1.0f, 0.0f),
		90.0f, width(), height(), 0.001f, 100.f
	);
	camera->setSpeed(0.5);

	particleObjShader = new ShaderProgram(this);
	particleObjShader->loadShaders(
		":/vertexShader",
		":/fragShader"
	);
	
	renderer = new Renderer();
	
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
	
	gameObjects = make_unique<std::vector<unique_ptr<GameObject>>>();

	m_sphereMesh = make_shared<Mesh>(Utils::readObj(QString(":/sphere")));
	m_sphereMaterial = make_shared<Material>(QString(":/blattTexture"));


	gameObjects->push_back(make_unique<GameObject>(
		true,
		100.f,
		m_sphereMesh,
		m_sphereMaterial
		)
	);
	gameObjects->push_back(make_unique<GameObject>(
		true,
		100.f,
		m_sphereMesh,
		m_sphereMaterial
		)
	);

}


void OGLWidget::paintGL() {

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	/*
	 render sun
	*/

	renderer->render(
		particleObjShader,
		*(gameObjects->at(0)),
		*(camera)
	);
	
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
	for (int i = 0; i < gameObjects->size(); i++) {
		gameObjects->at(i)->updateObject(frame, event);
	}
}

void OGLWidget::wheelEvent(QWheelEvent * event) {
	camera->update(event);
}
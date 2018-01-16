
#include "cercetare.hpp"

#include <QBuffer>
#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>

#include <functional>

Cercetare::Cercetare(QWidget *parent)
	: QMainWindow(parent)
{
	m_ui.setupUi(this);

	m_cl.SetLogFunction(std::bind(&Cercetare::Log, this, std::placeholders::_1));

	//m_ui.labelImageViewerOriginal->setHidden(true);

	// slots
	// File
	connect(m_ui.actionOpen_image, SIGNAL(triggered()), this, SLOT(OnLoadImageClicked()));

	// Edit
	connect(m_ui.actionGrayscale, SIGNAL(triggered()), this, SLOT(OnGrayscaleClicked()));


	connect(m_ui.actionSobel, SIGNAL(triggered()), this, SLOT(OnSobelClicked()));
	connect(m_ui.actionGaussian_Blur, SIGNAL(triggered()), this, SLOT(OnGaussianBlurClicked()));
	connect(m_ui.actionColor_Smoothing, SIGNAL(triggered()), this, SLOT(OnColorSmoothingClicked()));
	connect(m_ui.actionSharpening, SIGNAL(triggered()), this, SLOT(OnSharpeningClicked()));

	// Segmentation
	connect(m_ui.actionK_Means, SIGNAL(triggered()), this, SLOT(OnKMeansClicked()));
	connect(m_ui.actionThreshold, SIGNAL(triggered()), this, SLOT(OnThresholdClicked()));
	connect(m_ui.actionSOM, SIGNAL(triggered()), this, SLOT(OnSOMClicked()));
}

Cercetare::~Cercetare()
{

}

void Cercetare::Log(std::string str)
{
	m_ui.textBrowserLog->append(QString::fromStdString(str));
}


void Cercetare::LoadTexture(QString imgpath)
{
	// set image that should show
	////ui.openGLImageViewer->SetImage(imgpath);

	if (!m_img.load(imgpath)) {
		return;
	}

	m_pm = QPixmap::fromImage(m_img);

	m_ui.labelImageViewerOriginal->setPixmap(m_pm);
	m_ui.labelImageViewerResult->setPixmap(m_pm);

	m_cl.SetData(m_img);
}

// slots
void Cercetare::OnLoadImageClicked()
{
	const QString DEFAULT_DIR_KEY("default_dir");
	QSettings MySettings;

	QString filename = QFileDialog::getOpenFileName(
		this, "Select a file", MySettings.value(DEFAULT_DIR_KEY).toString());

	if (filename.isEmpty()) {
		return;
	}
	else {
		// save current dir
		QDir CurrentDir;
		MySettings.setValue(DEFAULT_DIR_KEY,
			CurrentDir.absoluteFilePath(filename));

		LoadTexture(filename);
	}
}

void Cercetare::OnSaveImageClicked()
{
	// TODO
}

void Cercetare::OnSobelClicked()
{
	m_cl.Sobel(m_img);
	m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void Cercetare::OnGaussianBlurClicked()
{
	m_cl.GaussianBlur(m_img);
	m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void Cercetare::OnSharpeningClicked()
{
	m_cl.Sharpening(m_img);
	m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void Cercetare::OnColorSmoothingClicked()
{
	m_cl.ColorSmoothing(m_img);
	m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void Cercetare::OnKMeansClicked()
{
	bool ok;
	int clusters_no = QInputDialog::getInt(this, tr("Clusters"),
		tr("Number of clusters:"), 0, 0, 20, 1, &ok);

	if (ok) {
		m_cl.KMeans(m_img, clusters_no);
		m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
	}
}

void Cercetare::OnGrayscaleClicked()
{
	m_cl.Grayscale(m_img);
	m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void Cercetare::OnThresholdClicked()
{
	bool ok;
	int value = QInputDialog::getInt(this, tr("Level"),
		tr("Threshold level:"), 0, 0, 255, 1, &ok);

	if (ok) {
		m_cl.Threshold(m_img, value / 255.f);
		m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
	}
}

void Cercetare::OnSOMClicked()
{
	const QString DEFAULT_DIR_KEY("default_dir");
	QSettings MySettings;

	QString filename = QFileDialog::getOpenFileName(
		this, "Select ground truth", MySettings.value(DEFAULT_DIR_KEY).toString());

	QImage *ground_truth = nullptr;
	if (filename.isEmpty()) {
		return;
	}
	else {
		// save current dir
		QDir CurrentDir;
		MySettings.setValue(DEFAULT_DIR_KEY,
			CurrentDir.absoluteFilePath(filename));

		ground_truth = new QImage;
		if (!ground_truth->load(filename)) {
			Log("Couldn't open ground truth file: " + filename.toStdString());
			delete ground_truth;
			ground_truth = nullptr;
		}
	}

	m_cl.SOMSegmentation(m_img, ground_truth);
	m_ui.labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}


#include "MainWindow.h"

#include <QBuffer>
#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>
#include <QScrollBar>
#include <QScreen>
#include <QWheelEvent>
#include <QGuiApplication>

#include <functional>

#include "Benchmark.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), m_scale_factor(1.0)
{
	m_ui.setupUi(this);
	m_ui.textBrowserLog->hide();

	scrollAreaViewImageOriginal = new QScrollArea(m_ui.centralWidget);
	scrollAreaViewImageResult = new QScrollArea(m_ui.centralWidget);
	labelImageViewerOriginal = new QLabel;
	labelImageViewerResult = new QLabel;

	m_cl.SetLogFunction(std::bind(&MainWindow::Log, this, std::placeholders::_1));
	
	labelImageViewerOriginal->setBackgroundRole(QPalette::Base);
	labelImageViewerOriginal->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	labelImageViewerOriginal->setScaledContents(true);

	labelImageViewerResult->setBackgroundRole(QPalette::Base);
	labelImageViewerResult->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	labelImageViewerResult->setScaledContents(true);

	//m_ui.scrollAreaViewImageResultWidgetContents->setScaledContents(true);


	scrollAreaViewImageOriginal->setBackgroundRole(QPalette::Dark);
	scrollAreaViewImageOriginal->setObjectName(QStringLiteral("scrollAreaViewImageOriginal"));
	scrollAreaViewImageOriginal->setWidget(labelImageViewerOriginal);
	scrollAreaViewImageOriginal->hide();

	scrollAreaViewImageResult->setBackgroundRole(QPalette::Dark);
	scrollAreaViewImageResult->setObjectName(QStringLiteral("scrollAreaViewImageResult"));
	scrollAreaViewImageResult->setWidget(labelImageViewerResult);

	m_ui.verticalLayout->insertWidget(1, scrollAreaViewImageOriginal);
	m_ui.verticalLayout->insertWidget(2, scrollAreaViewImageResult);


	SetActions();

	resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
}

MainWindow::~MainWindow()
{

}

void MainWindow::Log(std::string str)
{
	m_ui.textBrowserLog->append(QString::fromStdString(str));
}

void MainWindow::wheelEvent(QWheelEvent * e)
{
	// must rewrite scrollarea wheelevent
	//if (scrollAreaViewImageOriginal->underMouse() || scrollAreaViewImageResult->underMouse())
	//{
	//	if (e->modifiers().testFlag(Qt::ControlModifier))
	//	{
	//		if (e->delta() > 0) {
	//			OnZoomInClicked();
	//		}
	//		else if (e->delta() <0) {
	//			OnZoomOutClicked();
	//		}
	//
	//		e->ignore();
	//	}
	//}
}

void MainWindow::OnZoomInClicked()
{
	ScaleImage(1.25);
}

void MainWindow::OnZoomOutClicked()
{
	ScaleImage(0.8);
}

void MainWindow::OnZoomResetClicked()
{
	labelImageViewerOriginal->adjustSize();
	labelImageViewerResult->adjustSize();
	
	m_scale_factor = 1.0;

	m_ui.pushButtonZoomIn->setEnabled(true);
	m_ui.pushButtonZoomOut->setEnabled(true);
}


void MainWindow::LoadTexture(QString imgpath)
{
	if (!m_img.load(imgpath)) {
		return;
	}

	m_pm = QPixmap::fromImage(m_img);

	labelImageViewerOriginal->setPixmap(m_pm);
	labelImageViewerResult->setPixmap(m_pm);

	labelImageViewerOriginal->adjustSize();
	labelImageViewerResult->adjustSize();

	//m_ui.labelImageViewerOriginal->pixmap()->size().scale(sz, Qt::KeepAspectRatio);
	//m_ui.labelImageViewerResult->pixmap()->size().scale(sz, Qt::KeepAspectRatio);
	//
	//m_ui.labelImageViewerOriginal->resize(m_ui.labelImageViewerOriginal->pixmap()->size());
	//m_ui.labelImageViewerResult->resize(m_ui.labelImageViewerResult->pixmap()->size());

	m_cl.SetData(m_img);
}

void MainWindow::SetActions()
{
	// slots
	connect(m_ui.pushButtonSwitchView, SIGNAL(clicked()), this, SLOT(OnSwitchViewClicked()));
	connect(m_ui.pushButtonZoomIn, SIGNAL(clicked()), this, SLOT(OnZoomInClicked()));
	connect(m_ui.pushButtonZoomOut, SIGNAL(clicked()), this, SLOT(OnZoomOutClicked()));
	connect(m_ui.pushButtonZoomReset, SIGNAL(clicked()), this, SLOT(OnZoomResetClicked()));
	connect(m_ui.pushButtonToggleLog, SIGNAL(clicked()), this, SLOT(OnToggleLogClicked()));

	// File
	connect(m_ui.actionOpen_image, SIGNAL(triggered()), this, SLOT(OnLoadImageClicked()));
	connect(m_ui.actionSave_image, SIGNAL(triggered()), this, SLOT(OnSaveImageClicked()));

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

void MainWindow::ScaleImage(double factor)
{
	m_scale_factor *= factor;
	labelImageViewerOriginal->resize(m_scale_factor * labelImageViewerOriginal->pixmap()->size());
	labelImageViewerResult->resize(m_scale_factor * labelImageViewerResult->pixmap()->size());

	Log("Label size: " + std::to_string(labelImageViewerResult->size().width()) + " x " +
		std::to_string(labelImageViewerOriginal->size().height()));

	AdjustScrollBar(scrollAreaViewImageResult->horizontalScrollBar(), factor);
	AdjustScrollBar(scrollAreaViewImageResult->verticalScrollBar(), factor);

	m_ui.pushButtonZoomIn->setEnabled(m_scale_factor < 3.0);
	m_ui.pushButtonZoomOut->setEnabled(m_scale_factor > 0.333);
}

void MainWindow::AdjustScrollBar(QScrollBar * scrollbar, double factor)
{
	scrollbar->setValue(int(factor * scrollbar->value()
		+ ((factor - 1) * scrollbar->pageStep() / 2)));
}

void MainWindow::OnToggleLogClicked()
{
	m_ui.textBrowserLog->setHidden(!m_ui.textBrowserLog->isHidden());

	if (m_ui.textBrowserLog->isHidden())
	{
		m_ui.pushButtonToggleLog->setText("Show Log");
	}
	else
	{
		m_ui.pushButtonToggleLog->setText("Hide Log");
	}
}

// slots
void MainWindow::OnSwitchViewClicked()
{
	scrollAreaViewImageOriginal->setHidden(!scrollAreaViewImageOriginal->isHidden());
	scrollAreaViewImageResult->setHidden(!scrollAreaViewImageResult->isHidden());
}


void MainWindow::OnLoadImageClicked()
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

		OnZoomResetClicked();
		scrollAreaViewImageOriginal->setHidden(true);
		scrollAreaViewImageResult->setHidden(false);

		//Log(Benchmark::RunTests(m_cl, m_img));
	}
}

void MainWindow::OnSaveImageClicked()
{
	QString filename = QFileDialog::getSaveFileName(this, "Save file");
	bool saved = m_img.save(filename + ".png", nullptr, 100);
}

void MainWindow::OnSobelClicked()
{
	m_cl.Sobel(m_img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void MainWindow::OnGaussianBlurClicked()
{
	m_cl.GaussianBlur(m_img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void MainWindow::OnSharpeningClicked()
{
	m_cl.Sharpening(m_img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void MainWindow::OnColorSmoothingClicked()
{
	m_cl.ColorSmoothing(m_img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void MainWindow::OnKMeansClicked()
{
	bool ok;
	int clusters_no = QInputDialog::getInt(this, tr("Clusters"),
		tr("Number of clusters:"), 0, 0, 20, 1, &ok);

	if (ok) {
		m_cl.KMeans(m_img, clusters_no);
		labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
	}
}

void MainWindow::OnGrayscaleClicked()
{
	m_cl.Grayscale(m_img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void MainWindow::OnThresholdClicked()
{
	bool ok;
	int value = QInputDialog::getInt(this, tr("Level"),
		tr("Threshold level:"), 0, 0, 255, 1, &ok);

	if (ok) {
		m_cl.Threshold(m_img, value / 255.f);
		labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
	}
}

void MainWindow::OnSOMClicked()
{
	const QString DEFAULT_DIR_KEY("default_dir");
	QSettings MySettings;

	QString filename = QFileDialog::getOpenFileName(
		this, "Select ground truth", MySettings.value(DEFAULT_DIR_KEY).toString());

	QImage *ground_truth = nullptr;
	if (!filename.isEmpty()) {
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
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));


	delete ground_truth;
	ground_truth = nullptr;
}

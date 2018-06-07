
#include "MainWindow.h"

#include <QPainter>
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
#include "ConvolutionDialog.h"
#include "ShowImageDialog.h"
#include "SequentialImplementation.h"

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
	connect(m_ui.actionResize, SIGNAL(triggered()), this, SLOT(OnResizeClicked()));

	// FIlters
	connect(m_ui.actionSobel, SIGNAL(triggered()), this, SLOT(OnSobelClicked()));
	connect(m_ui.actionGaussian_Blur, SIGNAL(triggered()), this, SLOT(OnGaussianBlurClicked()));
	connect(m_ui.actionColor_Smoothing, SIGNAL(triggered()), this, SLOT(OnColorSmoothingClicked()));
	connect(m_ui.actionSharpening, SIGNAL(triggered()), this, SLOT(OnSharpeningClicked()));
	connect(m_ui.actionCustom_Convolution, SIGNAL(triggered()), this, SLOT(OnCustomConvolutionClicked()));
	

	// Segmentation
	connect(m_ui.actionK_Means, SIGNAL(triggered()), this, SLOT(OnKMeansClicked()));
	connect(m_ui.actionThreshold, SIGNAL(triggered()), this, SLOT(OnThresholdClicked()));
	connect(m_ui.actionSOM, SIGNAL(triggered()), this, SLOT(OnSOMClicked()));

	// Descriptors
	connect(m_ui.actionSIFT, SIGNAL(triggered()), this, SLOT(OnSIFTClcked()));
	connect(m_ui.actionFind_Image, SIGNAL(triggered()), this, SLOT(OnFindImageClicked()));

	// Help
	connect(m_ui.actionBenchmark, SIGNAL(triggered()), this, SLOT(OnBenchmarkClicked()));
	connect(m_ui.actionQuality_Tests, SIGNAL(triggered()), this, SLOT(OnQualityTestsClicked()));	
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

void MainWindow::OnCustomConvolutionClicked()
{
	ConvolutionDialog c_diag;
	c_diag.exec();

	std::vector<float> kernel_values = c_diag.GetKernel();
	if (true == kernel_values.empty())
	{
		return;
	}

	m_cl.CustomFilter(m_img, kernel_values);
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

void MainWindow::OnResizeClicked()
{

	m_cl.Resize(m_img);
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

	QImage img = m_img.copy();
	m_cl.SOMSegmentation(img, ground_truth);
	//SequentialImplementation si;
	//si.SetLogFunction(std::bind(&MainWindow::Log, this, std::placeholders::_1));
	//
	//si.SOMSegmentation(img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(img));


	delete ground_truth;
	ground_truth = nullptr;
}

void MainWindow::OnSIFTClcked()
{
	m_cl.RunSIFT(m_img);
	labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));
}

void MainWindow::OnFindImageClicked()
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

		QImage img;
		img.load(filename);

		ShowImageDialog dialog;
		dialog.SetImage(img);

		std::vector<float> rect = m_cl.FindImageSIFT(m_img, img);

		int left = rect[0] * m_img.width();
		int top = rect[2] * m_img.height();
		int width = (rect[1] - rect[0]) * m_img.width();
		int height = (rect[3] - rect[2]) * m_img.height();

		QPainter p;
		p.begin(&m_img);
		p.setPen(QPen(QColor(Qt::red)));
		p.setBrush(QBrush(QColor(Qt::red), Qt::NoBrush));
		p.drawRect(QRect(left, top, width, height));
		p.end();

		labelImageViewerResult->setPixmap(QPixmap::fromImage(m_img));

		dialog.exec();
	}
}

void MainWindow::OnBenchmarkClicked()
{
	Log(Benchmark::RunTests(m_cl, m_img));
}

void MainWindow::OnQualityTestsClicked()
{
	QImage imgGPUGrayscale, imgCPUGrayscale, imgGPUGaussianBlur, imgCPUGaussianBlur;
	imgGPUGrayscale = m_img.copy();
	imgCPUGrayscale = m_img.copy();

	imgGPUGaussianBlur = m_img.copy();
	imgCPUGaussianBlur = m_img.copy();

	m_cl.Grayscale(imgGPUGrayscale);
	m_cl.GaussianBlur(imgGPUGaussianBlur);

	SequentialImplementation si;

	si.Grayscale(imgCPUGrayscale);
	si.GaussianBlur(imgCPUGaussianBlur);

	std::vector<float> grayscale_values = m_cl.PSNR(imgGPUGrayscale, imgCPUGrayscale);
	std::vector<float> GaussianBlur_values = m_cl.PSNR(imgGPUGaussianBlur, imgCPUGaussianBlur);

	std::string res = "Grayscale\nMSE R, MSE G, MSE B, PSNR\n";
	for (float v : grayscale_values)
	{
		res += std::to_string(v) + ", ";
	}
	res += "\nGaussianBlur\nMSE R, MSE G, MSE B, PSNR\n";
	for (float v : GaussianBlur_values)
	{
		res += std::to_string(v) + ", ";
	}

	imgGPUGrayscale.save("D:\\workspace\\sift tests\\GPU_grayscale.png", nullptr, 100);
	imgGPUGaussianBlur.save("D:\\workspace\\sift tests\\GPU_GaussianBlur.png", nullptr, 100);
	imgCPUGrayscale.save("D:\\workspace\\sift tests\\CPU_grayscale.png", nullptr, 100);
	imgCPUGaussianBlur.save("D:\\workspace\\sift tests\\CPU_GaussianBlur.png", nullptr, 100);

	Log(res);
}

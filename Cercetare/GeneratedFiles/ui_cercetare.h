/********************************************************************************
** Form generated from reading UI file 'cercetare.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CERCETARE_H
#define UI_CERCETARE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CercetareClass
{
public:
    QAction *actionOpen_image;
    QAction *actionSave_image;
    QAction *actionSobel;
    QAction *actionGaussian_Blur;
    QAction *actionSharpening;
    QAction *actionColor_Smoothing;
    QAction *actionK_Means;
    QAction *actionThreshold;
    QAction *actionGrayscale;
    QAction *actionSOM;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QSplitter *splitter;
    QLabel *labelImageViewerOriginal;
    QLabel *labelImageViewerResult;
    QHBoxLayout *horizontalLayout;
    QTextBrowser *textBrowserLog;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuFilters;
    QMenu *menuSegmentation;
    QMenu *menuEdit;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *CercetareClass)
    {
        if (CercetareClass->objectName().isEmpty())
            CercetareClass->setObjectName(QStringLiteral("CercetareClass"));
        CercetareClass->resize(996, 598);
        actionOpen_image = new QAction(CercetareClass);
        actionOpen_image->setObjectName(QStringLiteral("actionOpen_image"));
        actionSave_image = new QAction(CercetareClass);
        actionSave_image->setObjectName(QStringLiteral("actionSave_image"));
        actionSobel = new QAction(CercetareClass);
        actionSobel->setObjectName(QStringLiteral("actionSobel"));
        actionGaussian_Blur = new QAction(CercetareClass);
        actionGaussian_Blur->setObjectName(QStringLiteral("actionGaussian_Blur"));
        actionSharpening = new QAction(CercetareClass);
        actionSharpening->setObjectName(QStringLiteral("actionSharpening"));
        actionColor_Smoothing = new QAction(CercetareClass);
        actionColor_Smoothing->setObjectName(QStringLiteral("actionColor_Smoothing"));
        actionK_Means = new QAction(CercetareClass);
        actionK_Means->setObjectName(QStringLiteral("actionK_Means"));
        actionThreshold = new QAction(CercetareClass);
        actionThreshold->setObjectName(QStringLiteral("actionThreshold"));
        actionGrayscale = new QAction(CercetareClass);
        actionGrayscale->setObjectName(QStringLiteral("actionGrayscale"));
        actionSOM = new QAction(CercetareClass);
        actionSOM->setObjectName(QStringLiteral("actionSOM"));
        centralWidget = new QWidget(CercetareClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        labelImageViewerOriginal = new QLabel(splitter);
        labelImageViewerOriginal->setObjectName(QStringLiteral("labelImageViewerOriginal"));
        splitter->addWidget(labelImageViewerOriginal);
        labelImageViewerResult = new QLabel(splitter);
        labelImageViewerResult->setObjectName(QStringLiteral("labelImageViewerResult"));
        splitter->addWidget(labelImageViewerResult);

        verticalLayout->addWidget(splitter);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        textBrowserLog = new QTextBrowser(centralWidget);
        textBrowserLog->setObjectName(QStringLiteral("textBrowserLog"));

        horizontalLayout->addWidget(textBrowserLog);


        verticalLayout->addLayout(horizontalLayout);

        CercetareClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(CercetareClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 996, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuFilters = new QMenu(menuBar);
        menuFilters->setObjectName(QStringLiteral("menuFilters"));
        menuSegmentation = new QMenu(menuBar);
        menuSegmentation->setObjectName(QStringLiteral("menuSegmentation"));
        menuEdit = new QMenu(menuBar);
        menuEdit->setObjectName(QStringLiteral("menuEdit"));
        CercetareClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(CercetareClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        CercetareClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(CercetareClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        CercetareClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuEdit->menuAction());
        menuBar->addAction(menuFilters->menuAction());
        menuBar->addAction(menuSegmentation->menuAction());
        menuFile->addAction(actionOpen_image);
        menuFile->addAction(actionSave_image);
        menuFilters->addAction(actionSobel);
        menuFilters->addAction(actionGaussian_Blur);
        menuFilters->addAction(actionSharpening);
        menuFilters->addAction(actionColor_Smoothing);
        menuSegmentation->addAction(actionK_Means);
        menuSegmentation->addAction(actionThreshold);
        menuSegmentation->addAction(actionSOM);
        menuEdit->addAction(actionGrayscale);

        retranslateUi(CercetareClass);

        QMetaObject::connectSlotsByName(CercetareClass);
    } // setupUi

    void retranslateUi(QMainWindow *CercetareClass)
    {
        CercetareClass->setWindowTitle(QApplication::translate("CercetareClass", "Cercetare", Q_NULLPTR));
        actionOpen_image->setText(QApplication::translate("CercetareClass", "Open image", Q_NULLPTR));
        actionSave_image->setText(QApplication::translate("CercetareClass", "Save image", Q_NULLPTR));
        actionSobel->setText(QApplication::translate("CercetareClass", "Sobel", Q_NULLPTR));
        actionGaussian_Blur->setText(QApplication::translate("CercetareClass", "Gaussian Blur", Q_NULLPTR));
        actionSharpening->setText(QApplication::translate("CercetareClass", "Sharpening", Q_NULLPTR));
        actionColor_Smoothing->setText(QApplication::translate("CercetareClass", "Color Smoothing", Q_NULLPTR));
        actionK_Means->setText(QApplication::translate("CercetareClass", "K-Means", Q_NULLPTR));
        actionThreshold->setText(QApplication::translate("CercetareClass", "Threshold", Q_NULLPTR));
        actionGrayscale->setText(QApplication::translate("CercetareClass", "Grayscale", Q_NULLPTR));
        actionSOM->setText(QApplication::translate("CercetareClass", "SOM", Q_NULLPTR));
        labelImageViewerOriginal->setText(QString());
        labelImageViewerResult->setText(QString());
        menuFile->setTitle(QApplication::translate("CercetareClass", "File", Q_NULLPTR));
        menuFilters->setTitle(QApplication::translate("CercetareClass", "Filters", Q_NULLPTR));
        menuSegmentation->setTitle(QApplication::translate("CercetareClass", "Segmentation", Q_NULLPTR));
        menuEdit->setTitle(QApplication::translate("CercetareClass", "Edit", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class CercetareClass: public Ui_CercetareClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CERCETARE_H

#pragma once
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QTableWidget>
#include <QPushButton>

#include <vector>

class ConvolutionDialog :
	public QDialog
{
	Q_OBJECT

public:
	ConvolutionDialog();
	virtual ~ConvolutionDialog();

	std::vector<float> GetKernel();

private slots:
	void KernelSizeChanged();
	void OnSubmitButtonClicked();


private:
	const uint16_t DEFAULT_KERNEL_SIZE = 3;

	uint16_t m_kernel_dim;

	QLabel *m_label_kernel_size;
	QLineEdit *m_input_kernel_size;
	QTableWidget *m_input_kernel;
	QPushButton *m_button_submit;

	std::vector<float> m_kernel_values;
};



#include "ConvolutionDialog.h"

#include <QValidator>
#include <QBoxLayout>
#include <QMessageBox>

#include "NumberDelegate.h"

ConvolutionDialog::ConvolutionDialog()
{
	m_kernel_dim = DEFAULT_KERNEL_SIZE;

	m_label_kernel_size = new QLabel(tr("Kernel size:"));
	m_input_kernel_size = new QLineEdit;
	m_input_kernel_size->setValidator(new QIntValidator(0, 999, this));
	m_label_kernel_size->setBuddy(m_input_kernel_size);

	connect(m_input_kernel_size, SIGNAL(textChanged(const QString &)), this, SLOT(KernelSizeChanged()));

	m_input_kernel = new QTableWidget;
	m_input_kernel->setColumnCount(m_kernel_dim);
	m_input_kernel->setRowCount(m_kernel_dim);
	m_input_kernel->setMinimumHeight(480);
	m_input_kernel->setMinimumWidth(600);
	m_input_kernel->setItemDelegate(new NumberDelegate);
	
	m_button_submit = new QPushButton(tr("Submit"));
	m_button_submit->setDefault(true);
	connect(m_button_submit, SIGNAL(clicked()), this, SLOT(OnSubmitButtonClicked()));

	QHBoxLayout *input_layout = new QHBoxLayout;
	input_layout->setMargin(0);
	input_layout->addWidget(m_label_kernel_size);
	input_layout->addWidget(m_input_kernel_size);

	QSpacerItem *spacer = new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Fixed);
	QHBoxLayout *submit_layout = new QHBoxLayout;
	submit_layout->setMargin(0);
	submit_layout->addWidget(m_button_submit);
	submit_layout->addSpacerItem(spacer);

	QVBoxLayout *main_layout = new QVBoxLayout;
	main_layout->addLayout(input_layout);
	main_layout->addWidget(m_input_kernel);
	main_layout->addLayout(submit_layout);

	setLayout(main_layout);
	setWindowTitle(tr("Custom Convolution"));
	setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
}
ConvolutionDialog::~ConvolutionDialog()
{
}

std::vector<float> ConvolutionDialog::GetKernel()
{
	return m_kernel_values;
}

void ConvolutionDialog::KernelSizeChanged()
{
	m_kernel_dim = m_input_kernel_size->text().toUInt();

	m_input_kernel->setColumnCount(m_kernel_dim);
	m_input_kernel->setRowCount(m_kernel_dim);
}

void ConvolutionDialog::OnSubmitButtonClicked()
{
	m_kernel_values.resize(m_kernel_dim * m_kernel_dim + 1);
	m_kernel_values[0] = m_kernel_dim;

	uint32_t idx = 1;
	for (uint16_t i = 0; i < m_kernel_dim; ++i)
	{
		for (uint16_t j = 0; j < m_kernel_dim; ++j)
		{
			auto item = m_input_kernel->item(i, j);

			if (!item || item->text().isEmpty())
			{
				m_kernel_values.clear();

				QMessageBox messageBox;
				messageBox.critical(0, "Error", "There are empty cells!");
				messageBox.setFixedSize(500, 200);

				return;
			}
			else
			{
				m_kernel_values[idx++] = item->text().toFloat();
			}
		}
	}

	close();
}
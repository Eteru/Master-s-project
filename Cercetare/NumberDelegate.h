#pragma once

#include <QItemDelegate>
#include <QLineEdit>

class NumberDelegate
	: public QItemDelegate
{
public:
	QWidget* createEditor(QWidget *parent, const QStyleOptionViewItem & option,
		const QModelIndex & index) const
	{
		QLineEdit *lineEdit = new QLineEdit(parent);
		// Set validator
		QDoubleValidator *validator = new QDoubleValidator(0.0, 999.0, 6, lineEdit);
		lineEdit->setValidator(validator);
		return lineEdit;
	}
};


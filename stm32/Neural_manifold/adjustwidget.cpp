#include "adjustwidget.h"
#include "ui_adjustwidget.h"

#include <QDebug>
#include <QListView>
#include <QLineEdit>
#include <QCollator>

AdjustWidget::AdjustWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::AdjustWidget)
{
    ui->setupUi(this);
    ui->comboBox_paramSelect->setView(new QListView());
    ui->comboBox_paramSelect->setEditable(true);
    ui->comboBox_paramSelect->lineEdit()->setAlignment(Qt::AlignCenter);
    ui->comboBox_paramSelect->lineEdit()->setStyleSheet("\
                                                        background:transparent;\
                                                        border-width:0;\
                                                        border-radius:0px;\
                                                        border-color:#3E3A39;\
                                                        border-bottom-width:1px;\
                                                        ");

    ui->comboBox_paramSelect->lineEdit()->setFocusPolicy(Qt::ClickFocus);

    connect(ui->comboBox_paramSelect->lineEdit(),&QLineEdit::editingFinished,
            this,&AdjustWidget::on_comboBox_textEditingFinished);
}

AdjustWidget::~AdjustWidget()
{
    delete ui;
}

void AdjustWidget::setTitel(QString titel)
{
    ui->label_title->setText(titel);
}

void AdjustWidget::setItems(QStringList &list,int current_index)
{
    ui->comboBox_paramSelect->clear();
    ui->comboBox_paramSelect->addItems(list);
    if(current_index < list.size()){
        ui->comboBox_paramSelect->setCurrentIndex(current_index);
    }
}

void AdjustWidget::selectFristItem()
{
    ui->comboBox_paramSelect->setCurrentIndex(0);
}

void AdjustWidget::selectLastItem()
{
    ui->comboBox_paramSelect->setCurrentIndex(ui->comboBox_paramSelect->count()-1);
}

void AdjustWidget::setLineEditMaxLength(int length)
{
    ui->comboBox_paramSelect->lineEdit()->setMaxLength(length);
}

void AdjustWidget::sortItem()
{
    int item_count = ui->comboBox_paramSelect->count();
    QString current_str = getCurrentItem();
    QStringList list;
    for(int index=0; index < item_count; index++){
        list.append(ui->comboBox_paramSelect->itemText(index));
    }

    //list 排序
    QCollator sorter;
    sorter.setNumericMode(true); // 启用数字排序模式
    sorter.setCaseSensitivity(Qt::CaseInsensitive); // 设置区分大小写模式
    std::sort(list.begin(),
              list.end(),
              [&](const QString& a, const QString& b) {
                return sorter.compare( a, b ) < 0;
    });

    qDebug() << "list" << list;

    //重新设置
    ui->comboBox_paramSelect->clear();
    ui->comboBox_paramSelect->addItems(list);
    ui->comboBox_paramSelect->setCurrentText(current_str);
}

void AdjustWidget::addItem(QString &arg)
{
    ui->comboBox_paramSelect->addItem(arg);
    sortItem();
}

QString AdjustWidget::getCurrentItem()
{
    return ui->comboBox_paramSelect->currentText();
}

int AdjustWidget::getCurrentItemIndex()
{
    return ui->comboBox_paramSelect->currentIndex();
}

void AdjustWidget::removeLastItem()
{
    ui->comboBox_paramSelect->removeItem(ui->comboBox_paramSelect->currentIndex());
}

void AdjustWidget::setLineEditReadOnly(bool en)
{
    ui->comboBox_paramSelect->lineEdit()->setReadOnly(en);
}

void AdjustWidget::setLineEditValidator(const QValidator *validator)
{
    ui->comboBox_paramSelect->lineEdit()->setValidator(validator);
}

void AdjustWidget::on_pushButton_adjustLower_clicked()
{
    int index = ui->comboBox_paramSelect->currentIndex();
    if(index > 0){
        ui->comboBox_paramSelect->setCurrentIndex(index-1);
    }
}

void AdjustWidget::on_pushButton_adjustUpper_clicked()
{
    int index = ui->comboBox_paramSelect->currentIndex();
    if(index < (ui->comboBox_paramSelect->count()-1)){
        ui->comboBox_paramSelect->setCurrentIndex(index+1);
    }
}

void AdjustWidget::on_comboBox_paramSelect_currentIndexChanged(const QString &arg1)
{
    on_itemChange(arg1);
}

void AdjustWidget::on_comboBox_textEditingFinished()
{
    if(ui->comboBox_paramSelect->hasFocus()){
        emit on_textEditingFinished(this,ui->comboBox_paramSelect->currentText());
        ui->comboBox_paramSelect->clearFocus();
    }
}

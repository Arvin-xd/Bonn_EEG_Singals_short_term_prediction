#ifndef ADJUSTWIDGET_H
#define ADJUSTWIDGET_H

#include <QWidget>
#include <QStringList>
#include <QIntValidator>

namespace Ui {
    class AdjustWidget;
}

enum AdjustWidgetEvent{
    ADJUST_EVENT_REDUCE,
    ADJUST_EVENT_ADD,
    ADJUST_EVENT_EDIT_COMPLETED,
};

class AdjustWidget : public QWidget
{
    Q_OBJECT

public:
    explicit AdjustWidget(QWidget *parent = nullptr);
    ~AdjustWidget();

    void setTitel(QString);
    void setItems(QStringList &list,int current_index);
    void selectFristItem();
    void selectLastItem();
    void setLineEditMaxLength(int length);

    void addItem(QString&);
    void sortItem(void);

    QString getCurrentItem();
    int getCurrentItemIndex();
    void removeLastItem();

    void setLineEditReadOnly(bool);
    void setLineEditValidator(const QValidator*);

    static const int ADJUSTWIDGET_ID_FREQUENCY      = 0x8001;
    static const int ADJUSTWIDGET_ID_VOLTAGE        = 0x8001;
    static const int ADJUSTWIDGET_ID_TYPES_OF_WAVE  = 0x8001;
    static const int ADJUSTWIDGET_ID_BURST_COUNT    = 0x8001;


signals:
    void on_itemChange(const QString &);
    void on_textEditingFinished(QObject*,const QString &);

private slots:
    void on_pushButton_adjustLower_clicked();
    void on_pushButton_adjustUpper_clicked();
    void on_comboBox_paramSelect_currentIndexChanged(const QString &arg1);
    void on_comboBox_textEditingFinished();

private:
    Ui::AdjustWidget *ui;
    int widget_id;

};

#endif // ADJUSTWIDGET_H

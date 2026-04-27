#ifndef CHANNELSELECTCRIDLAYOUT_H
#define CHANNELSELECTCRIDLAYOUT_H

#include <QObject>
#include <QGridLayout>
#include <QPushButton>
#include <QSignalMapper>

class ChannelSelectLayout: public QGridLayout
{
    Q_OBJECT
public:
    ChannelSelectLayout();
    ~ChannelSelectLayout();
    void setValue(int row,int col,float value);
private:
    QSignalMapper *signelMapper;
    QPushButton *btns[400][20];
    QList<int> *channelSelectList;
    int last_position;
public slots:
    void onChannalSelectButtonClick(int channel);

signals:
    void updateDataPosition(int row,int col);
};

#endif // CHANNELSELECTCRIDLAYOUT_H

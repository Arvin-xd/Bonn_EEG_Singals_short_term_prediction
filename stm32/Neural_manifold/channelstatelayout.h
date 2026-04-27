#ifndef CHANNELSTATELAYOUT_H
#define CHANNELSTATELAYOUT_H

#include <QLabel>
#include <QGridLayout>

class ChannelStateLayout: public QGridLayout
{
    Q_OBJECT
public:
    ChannelStateLayout();
    ~ChannelStateLayout();

    void setChannelState(int channel,bool state/*open:@true close:@false*/);
    void clear();
private:
    bool channelStates[10][8];
    QLabel *lbs[10][8];

    QString getLabelStyle(bool state);

signals:

};

#endif // CHANNELSTATELAYOUT_H

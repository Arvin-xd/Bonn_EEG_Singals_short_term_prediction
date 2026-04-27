#ifndef LOADDATA_H
#define LOADDATA_H

#include <QString>




class LoadData
{
public:
    LoadData();

    int loadCsvData(const QString &filePath, float *input_data,int row, int col);

};

#endif // LOADDATA_H

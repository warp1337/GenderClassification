#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showMat(const cv::Mat& in)
{
    cv::Mat img;
    cv::cvtColor(in, img,CV_BGR2RGB );
    QImage Qframe=QImage((const uchar*)img.data, img.cols,img.rows,img.step,QImage::Format_RGB888);
    ui->label->setPixmap(QPixmap::fromImage( Qframe.scaled(ui->label->width(),ui->label->height(),
                                                           Qt::KeepAspectRatio,Qt::FastTransformation)));
    ui->label_2->setText("waht");
}

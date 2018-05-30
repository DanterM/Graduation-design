# import tkinter
# top = tkinter.Tk()
# label = tkinter.Label(top,text='Hello World!')
# label.pack()
# tkinter.mainloop()


from PyQt5.QtWidgets import QApplication, QWidget
import sys
from PyQt5.QtGui import QIcon


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(1000, 800)
    w.move(100, 100)
    w.setWindowTitle('原型系统')
    w.show()


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('web.png'))

        self.show()

    sys.exit(app.exec_())
import tkinter
import tkinter.filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageFile
from OMR2.Restore import *
'''
这个是OMR的GUI界面

两个功能：
    1：选择文件按钮
    2：识别按钮
    按钮功能说明：
        选择文件按钮：
            1.选择后获取到路径名，但不保存
            2.显示
        识别按钮：
            根据路径名，将文件copy到项目中，然后调用识别方法
'''

# Restore的全局属性

# 自己的全局属性
top = tkinter.Tk()
top.title('OptionMusicRecognize')
# 长宽
top.geometry('640x540')
img_path = ''
downloadImg_path = './downloadImg/'
# Result_text = ""
Result_text = tkinter.StringVar()
Result_text.set('识别结果为:')

def choose_file_show():
    global img_path
    img_path = tkinter.filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser('f:/新建文件夹')))
    print('打开文件：', img_path)
    #检查文件：1是否为空 2：是否为jpg或者png后缀
    if img_path is not None:
        houzhui = img_path.split('.')[-1]
        print(houzhui)
        # png不确定是否会出错
        if houzhui.lower() == 'jpg' or houzhui.lower() == 'png':
            showImg(img_path)
            e.set(img_path)
            return
        elif img_path == "":
            return
        else:
            messagebox.showinfo(title='错误', message="文件错误,请选择jpg或png格式图片!")
    else:
        messagebox.showinfo(title='错误', message="文件为空")

load0 = Image.open("./MyImg/73.jpg")
render0 = ImageTk.PhotoImage(load0)
img = tkinter.Label(image=render0)
def showImg(imgpath):
    load = Image.open(imgpath)
    render = ImageTk.PhotoImage(load)
    # img = tkinter.Label(image=render)
    img.configure(image=render)
    img.place(x=0, y=100)
    img.image = render

e = tkinter.StringVar()

select_button = tkinter.Button(top, text="选择文件", command=choose_file_show).pack()
e_entry = tkinter.Entry(top, width=68, textvariable=e).pack(side=tkinter.TOP)
submit_button = tkinter.Button(top, text="识别", command=lambda: img_recognize(img_path)).pack()
showResult_label = tkinter.Label(top, textvariable=Result_text).pack()


def img_recognize(img_path):
    '''
    识别按钮
    就不需要判断了，选择文件的时候已经判断过了
    '''

    '''
    下面是把上传的图片移动到项目路径下
    暂时先不用，就直接根据图片所在磁盘地址进行识别
    '''
    # if img_path is not None:
    #     shutil.copy(src=img_path, dst=downloadImg_path)
    # print("源地址:", img_path)
    # print("下载到项目地址:", downloadImg_path)
    ## 获取到本地的图片地址
    # new_img_path = downloadImg_path+img_path.split('/')[-1]
    if img_path == '' or img_path is None:
        messagebox.showinfo(title='提示', message="未选择文件！")
        return
        # 1.调用
    pitch_name = recognize(img_path)

    # 2.显示结果在界面
    global Result_text
    Result_text.set('识别结果为:' + pitch_name)



# lambda用在这里是因为 command的参数是一个函数名，是不加括号的，但这样就不能传参了，所以加lambda
# show_button = tkinter.Button(top, text="显示图片", command=lambda: showImg(img_path))
# show_button.pack()
ImageFile.LOAD_TRUNCATED_IMAGES = True

top.mainloop()

import ast
from django.shortcuts import render
from .img import AllImg
def runoob(request):
    views_list = ["菜鸟教程1","菜鸟教程2","菜鸟教程3"]
    return render(request, "runoob.html", {"views_list": views_list})
from django.shortcuts import render
from django.views.decorators import csrf
 
# 接收POST请求数据
def post(request):
    ctx ={}
    if request.POST:
        ctx['rlt'] = request.POST['q']
    return render(request, "post.html", ctx)
def run_img(request):
    ctx = {}
    if request.POST:
        path = request.POST["path"]
        save_path = request.POST["savepath"]
        size = int(request.POST["size"])
        light_sum = int(request.POST["light_sum"])
        limit = request.POST["limit"]
        limit = ast.literal_eval(limit)
        file_paths = []
        for i in range(1,10001,1):
            file_paths.append(f"{path}/VimbaImage_{i}.tiff")
        save_path = f"{path}/{save_path}"
        print(f"{path}")

        ALLIMG = AllImg(file_paths = file_paths,light_sum = light_sum,save_path = save_path,limit = limit, size = size)
        ALLIMG.work()
        #limit = int(request.POST["limit"])
        ctx['rlt'] = "完成"
    return render(request, "run_img.html", ctx)
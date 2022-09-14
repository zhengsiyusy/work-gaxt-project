@torch.no_grad()# 该标注使得方法中所有计算得出的tensor的requires_grad都自动设置为False，也就是说不会求梯度，可以加快预测效率，减小资源消耗
def run(
        weights=ROOT / 'yolov5s.pt',  # 事先训练完成的权重文件，比如yolov5s.pt,假如使用官方训练好的文件（比如yolov5s）,则会自动下载
        source=ROOT / 'data/images',  # 预测时的输入数据，可以是文件/路径/URL/glob, 输入是0的话调用摄像头作为输入
        data=ROOT / 'data/coco128.yaml',  # 数据集文件
        imgsz=(640, 640),  # 预测时的放缩后图片大小(因为YOLO算法需要预先放缩图片), 两个值分别是height, width
        conf_thres=0.25,  # 置信度阈值, 高于此值的bounding_box才会被保留
        iou_thres=0.45,  # IOU阈值,高于此值的bounding_box才会被保留
        max_det=1000,  # 一张图片上检测的最大目标数量
        device='',  # 所使用的GPU编号，如果使用CPU就写cpu
        view_img=False,  # 是否在推理时预览图片
        save_txt=False,  # save results to *.txt 是否将结果保存在txt文件中
        save_conf=False,  # save confidences in --save-txt labels 是否将结果中的置信度保存在txt文件中
        save_crop=False,  # save cropped prediction boxes 是否保存裁剪后的预测框
        nosave=False,  # do not save images/videos 是否保存预测后的图片/视频
        classes=None,  # 过滤指定类的预测结果
        agnostic_nms=False,  # 如为True,则为class-agnostic. 否则为class-specific
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # 推理结果保存的路径
        name='exp',  # 结果保存文件夹的命名前缀
        exist_ok=False,  # True: 推理结果覆盖之前的结果 False: 推理结果新建文件夹保存,文件夹名递增
        line_thickness=3,  # 绘制Bounding_box的线宽度
        hide_labels=False,  # True: 隐藏标签
        hide_conf=False,  # True: 隐藏置信度
        half=False,  # use FP16 half-precision inference 是否使用半精度推理（节约显存）
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 是否需要保存图片,如果nosave(传入的参数)为false且source的结尾不是txt则保存图片 
    # 后面这个source.endswith('.txt')也就是source以.txt结尾，不过我不清楚这是什么用法
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是不是视频/图像文件路径
    # 假如source是"D://YOLOv5/data/1.jpg"，则Path(source).suffix是".jpg",Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))# 判断source是否是链接
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)# 判断是source是否是摄像头
    if is_url and is_file:
        source = check_file(source)  # 如果source是一个指向图片/视频的链接,则下载输入数据
        
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 根据前面生成的路径创建文件夹

    # 加载模型
    
    device = select_device(device)# select_device方法定义在utils.torch_utils模块中，返回值是torch.device对象，也就是推理时所使用的硬件资源。输入值如果是数字，表示GPU序号。也可是输入‘cpu’，表示使用CPU训练，默认是cpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)# DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    stride, names, pt = model.stride, model.names, model.pt 
    # stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
    # names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...] 
    # pt: 加载的是否是pytorch模型（也就是pt格式的文件），
    imgsz = check_img_size(imgsz, s=stride)  
    # 将图片大小调整为步长的整数倍
    # 比如假如步长是10，imagesz是[100,101],则返回值是[100,100]

    # Dataloader
    if webcam:# 使用摄像头作为输入
        view_img = check_imshow()# 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        cudnn.benchmark = True  # 该设置可以加速预测
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)# 加载输入数据流
        # source：输入数据源 image_size 图片识别前被放缩的大小， stride：识别时的步长， 
        # auto的作用可以看utils.augmentations.letterbox方法，它决定了是否需要将图片填充为正方形，如果auto=True则不需要
        bs = len(dataset)  # batch_size 批大小
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs# 用于保存视频,前者是视频路径,后者是一个cv2.VideoWriter对象
  # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # 使用空白图片（零矩阵）预先用GPU跑一遍预测流程，可以加速预测
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    # seen: 已经处理完了多少帧图片
    # windows: 如果需要预览图片,windows列表会给每个输入文件存储一个路径.
    # dt: 存储每一步骤的耗时
    for path, im, im0s, vid_cap, s in dataset:
    # 在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
    #path：文件路径（即source）
    #im: 处理后的输入图片列表（经过了放缩操作）
    #im0s: 源输入图片列表
    #vid_cap
    # s： 图片的基本信息，比如路径，大小
        t1 = time_sync()# 获取当前时间
        im = torch.from_numpy(im).to(device)#将图片放到指定设备(如GPU)上识别
        im = im.half() if model.fp16 else im.float()  # 把输入从整型转化为半精度/全精度浮点数。
        im /= 255  # 0 - 255 to 0.0 - 1.0 #将图片归一化处理（这是图像表示方法的的规范，使用浮点数就要归一化） 
        if len(im.shape) == 3:
            im = im[None]  # 添加一个第0维。在pytorch的nn.Module的输入中，第0维是batch的大小，这里添加一个1。
        t2 = time_sync() # 获取当前时间
        dt[0] += t2 - t1 # 记录该阶段耗时

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        # 如果为True则保留推理过程中的特征图，保存在runs文件夹中
        pred = model(im, augment=augment, visualize=visualize)
        # 推理结果，pred保存的是所有的bound_box的信息，
        t3 = time_sync()
        dt[1] += t3 - t2# 记录该阶段耗时

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # 执行非极大值抑制，返回值为过滤后的预测框
        # conf_thres： 置信度阈值
        # iou_thres： iou阈值
        # classes: 需要过滤的类（数字列表）
        # agnostic_nms： 标记class-agnostic或者使用class-specific方式。默认为class-agnostic
        # max_det: 检测框结果的最大数量
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # 每次迭代处理一张图片，
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #frame：此次取的是第几张图片
                s += f'{i}: '# s后面拼接一个字符串i
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # 推理结果图片保存的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 推理结果文本保存的路径
            s += '%gx%g ' % im.shape[2:]  # 显示推理前裁剪后的图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #得到原图的宽和高
            imc = im0.copy() if save_crop else im0  # for save_crop
            #如果save_crop的值为true， 则将检测到的bounding_box单独保存成一张图片。
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # 打印出所有的预测结果  比如1 person（检测出一个人）

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 保存txt文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        # 将坐标转变成x y w h 的形式，并归一化
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # line的形式是： ”类别 x y w h“，假如save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”

                    if save_img or save_crop or view_img:  # 给图片添加推理后的bounding_box边框
                        c = int(cls)  # 类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')# 类别名
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        #绘制边框
                        
                    if save_crop:# 将预测框内的图片单独保存
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
			#im0是绘制好的图片

            if view_img:# 如果view_img为true,则显示该图片
                if p not in windows: # 如果当前图片/视频的路径不在windows列表里,则说明需要重新为该图片/视频创建一个预览窗口
                    windows.append(p)# 标记当前图片/视频已经创建好预览窗口了
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) # 预览图片
                cv2.waitKey(1)  # 暂停 1 millisecond

            # Save results (image with detections)
            if save_img:# 如果save_img为true,则保存绘制完的图片
                if dataset.mode == 'image':# 如果是图片,则保存
                    cv2.imwrite(save_path, im0)
                else:  # 如果是视频或者"流"
                    if vid_path[i] != save_path:  # vid_path[i] != save_path,说明这张图片属于一段新的视频,需要重新创建视频文件
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                    # 以上的部分是保存视频文件

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')# 打印耗时
    t = tuple(x / seen * 1E3 for x in dt)  # 平均每张图片所耗费时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''# 标签保存的路径
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


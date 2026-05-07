from manim import *


FONT = "Microsoft YaHei"
BG = "#101410"
WHITE_SOFT = "#F3F7F4"
MUTED = "#C9D7CE"
GREEN = "#2E8F75"
ORANGE = "#E56F3C"
GOLD = "#E6B44F"
BLUE = "#4FB5E6"


class CNNForwardPass(Scene):
    """Clean seminar animation for the MNIST-V2.0 CNN forward pass."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("MNIST-V2.0 CNN 运行原理", font=FONT, font_size=34, color=WHITE_SOFT)
        title.to_edge(UP, buff=0.28)
        self.play(FadeIn(title, shift=DOWN))

        self.input_scene(title)
        self.convolution_scene(title)
        self.relu_pool_scene(title)
        self.second_block_scene(title)
        self.flatten_fc_scene(title)
        self.softmax_scene(title)
        self.binary_future_scene(title)

    def clear_scene(self, keep):
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob is not keep], run_time=0.45)

    def caption(self, text):
        cap = Text(text, font=FONT, font_size=23, color=MUTED)
        cap.to_edge(DOWN, buff=0.35)
        return cap

    def scene_heading(self, text):
        heading = Text(text, font=FONT, font_size=28, color=WHITE_SOFT)
        heading.move_to(UP * 2.35)
        return heading

    def make_digit_grid(self, cell_size=0.075):
        pixels = self.digit_pixels()
        cells = VGroup()
        for row in range(28):
            for col in range(28):
                value = pixels.get((row, col), 0.0)
                shade = int(18 + value * 225)
                square = Square(
                    side_length=cell_size,
                    stroke_width=0.15,
                    stroke_color="#45504A",
                    fill_color=rgb_to_color((shade / 255, shade / 255, shade / 255)),
                    fill_opacity=1,
                )
                square.move_to(RIGHT * ((col - 13.5) * cell_size) + DOWN * ((row - 13.5) * cell_size))
                cells.add(square)
        border = SurroundingRectangle(cells, color=ORANGE, buff=0.04, stroke_width=3)
        return VGroup(cells, border)

    def digit_pixels(self):
        pixels = {}

        def block(r0, r1, c0, c1, value=1.0):
            for row in range(r0, r1):
                for col in range(c0, c1):
                    pixels[(row, col)] = max(pixels.get((row, col), 0), value)

        block(5, 8, 7, 21)
        block(7, 14, 6, 10)
        block(12, 15, 7, 19)
        block(14, 21, 17, 21)
        block(20, 23, 8, 19)
        block(8, 10, 10, 13, 0.4)
        block(15, 17, 14, 17, 0.45)
        block(18, 21, 7, 10, 0.55)
        return pixels

    def make_kernel(self):
        kernel = VGroup()
        for row in range(3):
            for col in range(3):
                cell = Square(
                    side_length=0.36,
                    stroke_color=GOLD,
                    stroke_width=1.8,
                    fill_color=GOLD,
                    fill_opacity=0.12,
                )
                label = Text(f"w{row * 3 + col + 1}", font=FONT, font_size=15, color=WHITE_SOFT)
                group = VGroup(cell, label.move_to(cell))
                group.shift(RIGHT * col * 0.36 + DOWN * row * 0.36)
                kernel.add(group)
        kernel.center()
        return kernel

    def make_maps(self, count=4, color=GREEN, scale=1.0):
        maps = VGroup()
        for index in range(count):
            rect = RoundedRectangle(
                width=1.05,
                height=1.05,
                corner_radius=0.05,
                stroke_width=2,
                stroke_color=color,
                fill_color=color,
                fill_opacity=0.10 + 0.035 * index,
            )
            line1 = Line(rect.get_left() + RIGHT * 0.15, rect.get_right() + LEFT * 0.15, color=color, stroke_width=2)
            line2 = Line(rect.get_bottom() + UP * 0.15, rect.get_top() + DOWN * 0.15, color=color, stroke_width=2)
            VGroup(rect, line1, line2).shift(RIGHT * 0.18 * index + UP * 0.12 * index)
            maps.add(VGroup(rect, line1, line2))
        return maps.scale(scale)

    def make_stage(self, label, detail, color, width=2.2):
        box = RoundedRectangle(
            width=width,
            height=0.82,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=2.4,
            fill_color=color,
            fill_opacity=0.14,
        )
        text = Text(label, font=FONT, font_size=25, color=WHITE_SOFT).move_to(box)
        detail_text = Text(detail, font=FONT, font_size=19, color=MUTED).next_to(box, DOWN, buff=0.18)
        return VGroup(box, text, detail_text)

    def input_scene(self, title):
        heading = self.scene_heading("1. 输入：网页画板产生 784 个灰度值")

        grid = self.make_digit_grid(0.083).move_to(LEFT * 2.25 + DOWN * 0.1)
        grid_label = Text("28 x 28", font=FONT, font_size=25, color=WHITE_SOFT).next_to(grid, DOWN, buff=0.22)

        vector = VGroup()
        for i in range(34):
            rect = Rectangle(
                width=0.055,
                height=1.8,
                stroke_width=0.5,
                stroke_color=BLUE,
                fill_color=BLUE,
                fill_opacity=0.12 + 0.55 * ((i % 7) / 7),
            )
            rect.shift(RIGHT * i * 0.065)
            vector.add(rect)
        vector.move_to(RIGHT * 2.6 + DOWN * 0.1)
        vector_label = Text("像素向量 x：1 x 784", font=FONT, font_size=25, color=WHITE_SOFT)
        vector_label.next_to(vector, DOWN, buff=0.22)

        arrow = Arrow(grid.get_right(), vector.get_left(), buff=0.45, color=MUTED)
        cap = self.caption("用户在 Web 端画出的每个格子，都会成为神经网络输入中的一个数。")

        self.play(FadeIn(heading, shift=DOWN))
        self.play(FadeIn(grid), FadeIn(grid_label))
        self.play(GrowArrow(arrow), FadeIn(vector), FadeIn(vector_label))
        self.play(FadeIn(cap, shift=UP))
        self.wait(1.2)
        self.clear_scene(title)

    def convolution_scene(self, title):
        heading = self.scene_heading("2. 卷积：用小窗口扫描局部笔画")

        grid = self.make_digit_grid(0.074).move_to(LEFT * 4.2 + DOWN * 0.12)
        patch = Square(side_length=0.074 * 3.15, color=GOLD, stroke_width=4)
        patch.move_to(grid[0][5 * 28 + 7].get_center() + RIGHT * 0.074 + DOWN * 0.074)

        kernel = self.make_kernel().move_to(ORIGIN + DOWN * 0.1)
        kernel_label = Text("3 x 3 卷积核", font=FONT, font_size=23, color=GOLD).next_to(kernel, DOWN, buff=0.22)

        maps = self.make_maps(4, GREEN, 1.0).move_to(RIGHT * 4.0 + DOWN * 0.05)
        maps_label = Text("Conv1 输出：8 个特征图", font=FONT, font_size=23, color=WHITE_SOFT)
        maps_label.next_to(maps, DOWN, buff=0.28)

        arrow1 = Arrow(grid.get_right(), kernel.get_left(), buff=0.4, color=MUTED)
        arrow2 = Arrow(kernel.get_right(), maps.get_left(), buff=0.4, color=MUTED)
        cap = self.caption("卷积核关注局部区域，例如横线、竖线、拐角，输出多个特征图。")

        self.play(FadeIn(heading, shift=DOWN))
        self.play(FadeIn(grid), Create(patch))
        self.play(patch.animate.shift(RIGHT * 0.65), run_time=0.45)
        self.play(patch.animate.shift(DOWN * 0.6), run_time=0.45)
        self.play(GrowArrow(arrow1), FadeIn(kernel), FadeIn(kernel_label))
        self.play(GrowArrow(arrow2), FadeIn(maps), FadeIn(maps_label))
        self.play(FadeIn(cap, shift=UP))
        self.wait(1.2)
        self.clear_scene(title)

    def relu_pool_scene(self, title):
        heading = self.scene_heading("3. ReLU 和池化：筛掉弱响应，缩小特征图")

        maps1 = self.make_maps(4, GREEN, 0.92).move_to(LEFT * 4.3 + DOWN * 0.15)
        relu = self.make_stage("ReLU", "负数变 0", ORANGE, width=1.7).move_to(LEFT * 1.25 + DOWN * 0.05)
        pool = self.make_stage("MaxPool", "2 x 2 取最大值", GOLD, width=2.15).move_to(RIGHT * 1.55 + DOWN * 0.05)
        maps2 = self.make_maps(4, GOLD, 0.72).move_to(RIGHT * 4.55 + DOWN * 0.08)

        label1 = Text("28 x 28 特征图", font=FONT, font_size=21, color=MUTED).next_to(maps1, DOWN, buff=0.25)
        label2 = Text("14 x 14 特征图", font=FONT, font_size=21, color=MUTED).next_to(maps2, DOWN, buff=0.25)

        arrows = VGroup(
            Arrow(maps1.get_right(), relu.get_left(), buff=0.28, color=MUTED),
            Arrow(relu.get_right(), pool.get_left(), buff=0.28, color=MUTED),
            Arrow(pool.get_right(), maps2.get_left(), buff=0.28, color=MUTED),
        )
        cap = self.caption("ReLU 提供非线性；池化让模型对轻微位移更稳定，也减少后续计算量。")

        self.play(FadeIn(heading, shift=DOWN))
        self.play(FadeIn(maps1), FadeIn(label1))
        self.play(GrowArrow(arrows[0]), FadeIn(relu))
        self.play(GrowArrow(arrows[1]), FadeIn(pool))
        self.play(GrowArrow(arrows[2]), FadeIn(maps2), FadeIn(label2))
        self.play(FadeIn(cap, shift=UP))
        self.wait(1.2)
        self.clear_scene(title)

    def second_block_scene(self, title):
        heading = self.scene_heading("4. 第二个卷积块：组合出更高层的数字特征")

        stages = VGroup(
            self.make_stage("输入", "8 x 14 x 14", GREEN, width=1.65),
            self.make_stage("Conv2", "16 个卷积核", BLUE, width=1.8),
            self.make_stage("ReLU", "保留正响应", ORANGE, width=1.65),
            self.make_stage("MaxPool", "14 x 14 -> 7 x 7", GOLD, width=2.2),
            self.make_stage("输出", "16 x 7 x 7", GREEN, width=1.65),
        ).arrange(RIGHT, buff=0.55)
        stages.move_to(DOWN * 0.12)

        arrows = VGroup()
        for left, right in zip(stages[:-1], stages[1:]):
            arrows.add(Arrow(left.get_right(), right.get_left(), buff=0.12, color=MUTED, stroke_width=3))

        cap = self.caption("第一层更像找基础笔画；第二层会组合这些笔画，形成更接近数字结构的特征。")

        self.play(FadeIn(heading, shift=DOWN))
        self.play(FadeIn(stages[0]))
        for arrow, stage in zip(arrows, stages[1:]):
            self.play(GrowArrow(arrow), FadeIn(stage), run_time=0.55)
        self.play(FadeIn(cap, shift=UP))
        self.wait(1.2)
        self.clear_scene(title)

    def flatten_fc_scene(self, title):
        heading = self.scene_heading("5. 展平 + 全连接：把图像特征变成分类分数")

        maps = self.make_maps(5, GREEN, 0.78).move_to(LEFT * 4.2 + DOWN * 0.05)
        maps_label = Text("16 x 7 x 7", font=FONT, font_size=22, color=MUTED).next_to(maps, DOWN, buff=0.25)

        vector = VGroup()
        for i in range(38):
            rect = Rectangle(
                width=0.045,
                height=1.55,
                stroke_width=0.4,
                stroke_color=BLUE,
                fill_color=BLUE,
                fill_opacity=0.18 + 0.55 * ((i % 6) / 6),
            )
            rect.shift(RIGHT * i * 0.052)
            vector.add(rect)
        vector.move_to(LEFT * 0.25 + DOWN * 0.06)
        vector_label = Text("Flatten: 784 个特征", font=FONT, font_size=22, color=MUTED).next_to(vector, DOWN, buff=0.25)

        fc = self.make_fc_network().move_to(RIGHT * 3.75 + DOWN * 0.03)
        fc_label = Text("FC(10)", font=FONT, font_size=23, color=WHITE_SOFT).next_to(fc, DOWN, buff=0.25)

        arrow1 = Arrow(maps.get_right(), vector.get_left(), buff=0.38, color=MUTED)
        arrow2 = Arrow(vector.get_right(), fc.get_left(), buff=0.38, color=MUTED)
        cap = self.caption("展平后进入全连接层，最终得到 10 个数字类别的原始分数。")

        self.play(FadeIn(heading, shift=DOWN))
        self.play(FadeIn(maps), FadeIn(maps_label))
        self.play(GrowArrow(arrow1), FadeIn(vector), FadeIn(vector_label))
        self.play(GrowArrow(arrow2), FadeIn(fc), FadeIn(fc_label))
        self.play(FadeIn(cap, shift=UP))
        self.wait(1.2)
        self.clear_scene(title)

    def make_fc_network(self):
        layers = VGroup()
        specs = [(5, BLUE), (4, ORANGE), (6, GOLD)]
        for layer_index, (count, color) in enumerate(specs):
            layer = VGroup()
            for i in range(count):
                dot = Circle(radius=0.085, stroke_width=1.5, color=color, fill_color=color, fill_opacity=0.85)
                dot.shift(UP * ((count - 1) / 2 - i) * 0.34 + RIGHT * layer_index * 0.9)
                layer.add(dot)
            layers.add(layer)

        edges = VGroup()
        for left, right in zip(layers[:-1], layers[1:]):
            for a in left:
                for b in right:
                    edges.add(Line(a.get_center(), b.get_center(), color=MUTED, stroke_width=0.55, stroke_opacity=0.32))
        return VGroup(edges, layers)

    def softmax_scene(self, title):
        heading = self.scene_heading("6. Softmax：把 10 个分数转换成概率")

        probs = [0.01, 0.02, 0.03, 0.04, 0.06, 0.76, 0.02, 0.01, 0.03, 0.02]
        bars = VGroup()
        for digit, prob in enumerate(probs):
            track = Rectangle(width=0.33, height=2.25, stroke_color="#4C5650", stroke_width=1.2)
            fill = Rectangle(
                width=0.33,
                height=max(0.08, 2.25 * prob),
                stroke_width=0,
                fill_color=ORANGE if digit == 5 else GREEN,
                fill_opacity=0.92,
            )
            fill.align_to(track, DOWN)
            number = Text(str(digit), font=FONT, font_size=21, color=WHITE_SOFT).next_to(track, DOWN, buff=0.12)
            percent = Text(f"{prob * 100:.0f}%", font=FONT, font_size=16, color=MUTED).next_to(track, UP, buff=0.10)
            bars.add(VGroup(track, fill, number, percent).shift(RIGHT * digit * 0.62))
        bars.center().shift(DOWN * 0.15)

        result = Text("预测结果：5    置信度：76%", font=FONT, font_size=31, color=GOLD)
        result.next_to(bars, DOWN, buff=0.55)
        cap = self.caption("Web 页面右侧的概率柱状图，本质上就是这里的 softmax 输出。")

        self.play(FadeIn(heading, shift=DOWN))
        self.play(FadeIn(bars, lag_ratio=0.05))
        self.play(FadeIn(result, shift=UP), FadeIn(cap, shift=UP))
        self.wait(1.3)
        self.clear_scene(title)

    def binary_future_scene(self, title):
        new_title = Text("未来设想：二值化 CNN 推理", font=FONT, font_size=34, color=WHITE_SOFT)
        new_title.to_edge(UP, buff=0.28)
        self.play(Transform(title, new_title))

        float_box = self.future_box(
            "当前 float32 推理",
            ["权重和激活是浮点数", "核心计算：乘法 + 加法", "准确率高，但计算和存储更重"],
            BLUE,
        ).move_to(LEFT * 3.15 + DOWN * 0.1)
        binary_box = self.future_box(
            "未来 0/1 推理",
            ["权重和激活压缩到 1 bit", "核心计算：XNOR + bitcount", "目标：更小、更快、更适合边缘设备"],
            GOLD,
        ).move_to(RIGHT * 3.15 + DOWN * 0.1)
        arrow = Arrow(float_box.get_right(), binary_box.get_left(), buff=0.32, color=ORANGE, stroke_width=5)
        cap = self.caption("这个项目已经打通训练和部署链路，下一步就可以在推理层插入量化和二值化实验。")

        self.play(FadeIn(float_box, shift=RIGHT))
        self.play(GrowArrow(arrow), FadeIn(binary_box, shift=RIGHT))
        self.play(FadeIn(cap, shift=UP))
        self.wait(1.8)

    def future_box(self, title, lines, color):
        title_obj = Text(title, font=FONT, font_size=27, color=WHITE_SOFT)
        line_objs = VGroup(*[Text(line, font=FONT, font_size=21, color=MUTED) for line in lines])
        content = VGroup(title_obj, line_objs.arrange(DOWN, aligned_edge=LEFT, buff=0.2))
        content.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        box = RoundedRectangle(
            width=4.55,
            height=2.55,
            corner_radius=0.12,
            stroke_color=color,
            stroke_width=2.5,
            fill_color=color,
            fill_opacity=0.10,
        )
        content.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.35)
        return VGroup(box, content)

import tkinter
from tkinter.filedialog import askopenfilename, asksaveasfilename

import pygame
import numpy as np
from pygame.event import Event
from pygame.surface import Surface, SurfaceType
from queue import Queue

screen: Surface | SurfaceType = None
is_editing_enabled = True
is_rmb_down = False
last_point_pos = None
event_handlers = dict()
keypress_handlers = dict()
points = []
DBSCAN_STEP = pygame.USEREVENT + 1
DBSCAN_STOP = pygame.USEREVENT + 2


def event_handler(event_type: int):
    def decorator(func):
        event_handlers[event_type] = func
        return func

    return decorator


def setup():
    global screen

    pygame.init()
    screen = pygame.display.set_mode([800, 400])
    screen.fill(color='white')

    pygame.display.update()
    while True:
        loop()


def loop():
    for event in pygame.event.get():
        if event.type in event_handlers:
            print('Handling event: %s' % event)
            event_handlers[event.type](event)
    pygame.display.update()


@event_handler(pygame.QUIT)
def quit_handler(event: Event):
    pygame.quit()
    exit()


def add_point(pos):
    if not is_editing_enabled:
        return

    points.append(np.array(pos))
    pygame.draw.circle(screen, 'black', pos, 4)


@event_handler(pygame.MOUSEBUTTONDOWN)
def mouse_down_handler(event: Event):
    global is_rmb_down

    if event.button == pygame.BUTTON_LEFT:
        add_point(event.pos)
    if event.button == pygame.BUTTON_RIGHT:
        is_rmb_down = True


@event_handler(pygame.MOUSEBUTTONUP)
def mouse_up_handler(event: Event):
    global is_rmb_down, last_point_pos

    if event.button == pygame.BUTTON_RIGHT:
        is_rmb_down = False
        last_point_pos = None


@event_handler(pygame.MOUSEMOTION)
def mouse_motion_handler(event: Event):
    global last_point_pos

    if is_rmb_down:
        if last_point_pos is None or np.sum((event.pos - last_point_pos) ** 2) > 125:
            add_point(event.pos)
            last_point_pos = np.array(event.pos)


@event_handler(pygame.KEYDOWN)
def keypress_global_handler(event: Event):
    if event.key in keypress_handlers:
        keypress_handlers[event.key](event)


def keypress_handler(key_type: int):
    def decorator(func):
        keypress_handlers[key_type] = func
        return func

    return decorator


def clear_points():
    global points
    if not is_editing_enabled:
        return

    points = []
    screen.fill(color='white')


@keypress_handler(pygame.K_ESCAPE)
def escape_press_handler(event: Event):
    clear_points()


def load_data_from_file(path):
    if not path or not is_editing_enabled:
        return

    try:
        arr = list(np.loadtxt(path, delimiter=','))
        clear_points()
        for x, y in arr:
            add_point((x, y))
    except:
        pass


def save_data(path):
    if not path:
        return

    try:
        np.savetxt(path, np.array(points), delimiter=',')
    except:
        pass


@keypress_handler(pygame.K_l)
def load_press_handler(event: Event):
    if not is_editing_enabled:
        return

    tk = tkinter.Tk()
    tk.withdraw()  # hide window
    file_name = askopenfilename(parent=tk, defaultextension='.csv', filetypes=[('CSV', '*.csv')])
    tk.destroy()
    load_data_from_file(file_name)


@keypress_handler(pygame.K_s)
def save_press_handler(event: Event):
    tk = tkinter.Tk()
    tk.withdraw()  # hide window
    file_name = asksaveasfilename(parent=tk, defaultextension='.csv', filetypes=[('CSV', '*.csv')])
    tk.destroy()
    save_data(file_name)


@keypress_handler(pygame.K_RETURN)
def enter_press_handler(event: Event):
    start_dbscan(20, 2, auto=True)


@keypress_handler(pygame.K_SPACE)
def enter_press_handler(event: Event):
    evt = pygame.event.Event(DBSCAN_STEP)
    pygame.time.set_timer(evt, 50, loops=1)


def start_dbscan(radius=20, min_points=3, auto=True):
    global is_editing_enabled
    is_editing_enabled = False

    points_array = np.array(points)
    # -1 - not visited, 0 - in queue, 1 - core point
    # 2 - potential outlier, 3 - edge point, 4 - outlier
    flag_points_array = np.full(points_array.shape[0], -1)
    idx_array = np.arange(points_array.shape[0])
    cluster_points_array = np.full(points_array.shape[0], -1)
    cluster_color = []
    current_cluster = 0
    points_queue = Queue(points_array.shape[0])
    points_queue.put(0)
    event = pygame.event.Event(DBSCAN_STEP)
    pygame.event.post(event)
    prev_point = None

    def draw_points(curr=None):
        for idx in idx_array:
            pos = points_array[idx]
            cluster = cluster_points_array[idx]
            if flag_points_array[idx] == -1:
                pygame.draw.circle(screen, 'black', pos, 4)
            elif flag_points_array[idx] == 0:
                pygame.draw.circle(screen, 'black', pos, 4)
                pygame.draw.circle(screen, 'orange', pos, 1.5)
            elif flag_points_array[idx] == 2:
                pygame.draw.circle(screen, 'orange', pos, 4)
                pygame.draw.circle(screen, 'white', pos, 2)
            elif cluster != -1:
                pygame.draw.circle(screen, cluster_color[cluster], pos, 4)

            if flag_points_array[idx] == 3:
                pygame.draw.circle(screen, 'orange', pos, 8, 2)
        if curr is not None:
            pygame.draw.circle(screen, 'red', curr, 2)

    screen.fill(color='white')
    draw_points()

    @event_handler(DBSCAN_STEP)
    def dbscan_step_handler(event: Event):
        nonlocal current_cluster, prev_point
        point_idx = points_queue.get()
        point = points_array[point_idx]
        if prev_point is not None:
            pygame.draw.circle(screen, 'white', prev_point, radius, 1)
        pygame.draw.circle(screen, 'red', point, radius, 1)
        prev_point = point
        condition = np.sum((points_array - point) ** 2, axis=1) <= radius * radius
        condition[point_idx] = False
        neighbors = idx_array[condition]

        if len(neighbors) < min_points:
            if cluster_points_array[point_idx] == -1:
                flag_points_array[point_idx] = 2
            else:
                flag_points_array[point_idx] = 3
        else:
            flag_points_array[point_idx] = 1
            if cluster_points_array[point_idx] == -1:
                cluster_points_array[point_idx] = current_cluster
                cluster_color.append(tuple(np.random.choice(range(256), size=3)))
                current_cluster += 1
            for idx in neighbors:
                if flag_points_array[idx] == 0 or flag_points_array[idx] == 1:
                    continue
                cluster_points_array[idx] = cluster_points_array[point_idx]
                if flag_points_array[idx] == 2:
                    flag_points_array[idx] = 3
                else:
                    points_queue.put(idx)
                    flag_points_array[idx] = 0

        if points_queue.empty():
            next_points = np.where(flag_points_array == -1)[0]
            if len(next_points) != 0:
                points_queue.put(next_points[0])
            else:
                pygame.draw.circle(screen, 'white', point, radius, 1)
                draw_points()
                evt = pygame.event.Event(DBSCAN_STOP)
                pygame.event.post(evt)
                return

        draw_points(point)

        if auto:
            evt = pygame.event.Event(DBSCAN_STEP)
            pygame.time.set_timer(evt, 50, loops=1)

    @event_handler(DBSCAN_STOP)
    def dbscan_stop_handler(event: Event):
        global is_editing_enabled
        outliers = idx_array[flag_points_array == 2]
        for idx in outliers:
            flag_points_array[idx] = 4
            pygame.draw.circle(screen, 'red', points_array[idx], 4)
            pygame.draw.circle(screen, 'red', points_array[idx], 8, 2)
        is_editing_enabled = True


if __name__ == '__main__':
    setup()

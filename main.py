import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import glob
import time
import threading
from datetime import datetime
import numpy as np

# Проверяем наличие watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
    print("Watchdog доступен")
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Watchdog не установлен, используем периодическую проверку")

# Загрузка модели нейронной сети
try:
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
    print("TensorFlow доступен")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow не установлен")


class PhotoWatcher(FileSystemEventHandler):
    def __init__(self, app, folder_path):
        self.app = app
        self.folder_path = folder_path

    def on_created(self, event):
        """Вызывается при создании нового файла в папке"""
        self.handle_event(event)

    def on_moved(self, event):
        """Вызывается при перемещении файла в папку"""
        self.handle_event(event)

    def on_modified(self, event):
        """Вызывается при изменении файла"""
        self.handle_event(event)

    def handle_event(self, event):
        """Обрабатывает все файловые события"""
        src_path = getattr(event, 'src_path', None)
        dest_path = getattr(event, 'dest_path', None)
        file_path = dest_path if dest_path else src_path

        if not event.is_directory and self.is_image_file(file_path):
            print(f"Обнаружено изменение: {file_path}")
            # Добавляем задержку для полной записи файла
            self.app.root.after(1000, lambda: self.app.add_new_photo(file_path))

    def is_image_file(self, file_path):
        """Проверяет, является ли файл изображением"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')
        return file_path.lower().endswith(image_extensions)


class PhotoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Проверка на дефекты")

        # Полноэкранный режим с корректным определением рабочей области
        self.is_fullscreen = True

        # Используем state('zoomed') для правильного определения рабочей области
        try:
            self.root.state('zoomed')  # Это автоматически учитывает панель задач
        except:
            # Если не поддерживается, используем geometry с корректными размерами
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Получаем реальные размеры рабочей области
            self.root.update_idletasks()
            work_area_width = self.root.winfo_width()
            work_area_height = self.root.winfo_height()

            # Если не удалось получить рабочую область, используем эмпирические значения
            if work_area_width == 1:  # Tkinter иногда возвращает 1 до отображения окна
                work_area_width = screen_width
                work_area_height = screen_height - 40  # Предполагаемая высота панели задач

            self.root.geometry(f"{work_area_width}x{work_area_height}+0+0")

        # Начальный путь к папке с фотографиями
        self.photos_folder = ""

        # Сохраняем состояние watchdog как атрибут объекта
        self.WATCHDOG_AVAILABLE = WATCHDOG_AVAILABLE

        # Загрузка модели нейронной сети
        self.model = None
        if TENSORFLOW_AVAILABLE:
            try:
                self.model = load_model('defect_detection_continued.h5')
                print("✅ Модель нейронной сети загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
                self.model = None
        else:
            print("❌ TensorFlow недоступен, анализ будет выполняться в демо-режиме")

        # Переменные для управления состоянием
        self.photos = []
        self.current_photo_path = None
        self.current_photo_data = None
        self.known_files = set()
        self.analyzed_photos = {}
        self.current_photo_reference = None
        self.is_waiting_mode = False

        # Идентификаторы запланированных задач
        self.monitoring_after_id = None

        # Инициализация интерфейса
        self.create_main_menu()

        # Загрузка сохраненной папки если есть
        self.load_saved_folder()

    def get_work_area(self):
        """Возвращает размеры рабочей области (без панели задач)"""
        try:
            # Пробуем получить реальные размеры после отображения окна
            self.root.update_idletasks()

            # Получаем размеры окна в нормальном состоянии
            self.root.state('normal')
            self.root.geometry("800x600")
            self.root.update()

            normal_width = self.root.winfo_width()
            normal_height = self.root.winfo_height()

            # Переходим в полноэкранный режим и получаем размеры
            self.root.state('zoomed')
            self.root.update()

            zoomed_width = self.root.winfo_width()
            zoomed_height = self.root.winfo_height()

            print(f"Нормальный размер: {normal_width}x{normal_height}")
            print(f"Полноэкранный размер: {zoomed_width}x{zoomed_height}")

            return zoomed_width, zoomed_height

        except Exception as e:
            print(f"Ошибка при определении рабочей области: {e}")
            # Возвращаем размеры экрана с запасом для панели задач
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            return screen_width, screen_height - 50

    def create_main_menu(self):
        """Создает главное меню с выбором папки"""
        # Очищаем окно
        for widget in self.root.winfo_children():
            widget.destroy()

        # Основной фрейм
        main_frame = tk.Frame(self.root, bg='lightgray')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = tk.Label(
            main_frame,
            text="Проверка на дефекты",
            font=("Arial", 32, "bold"),
            bg='lightgray',
            fg='darkblue',
            justify=tk.CENTER
        )
        title_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

        # Информация о модели
        model_status = "✅ Модель нейронной сети загружена" if self.model else "❌ Модель недоступна (демо-режим)"
        model_label = tk.Label(
            main_frame,
            text=model_status,
            font=("Arial", 14),
            bg='lightgray',
            fg='green' if self.model else 'red',
            justify=tk.CENTER
        )
        model_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        # Кнопка выбора папки
        select_folder_button = tk.Button(
            main_frame,
            text="Выбрать папку с фото",
            command=self.select_folder,
            font=("Arial", 20),
            bg='green',
            fg='white',
            width=25,
            height=2
        )
        select_folder_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        # Кнопка "Начать проверку"
        self.start_button = tk.Button(
            main_frame,
            text="Начать проверку",
            command=self.start_viewing,
            font=("Arial", 20),
            bg='blue',
            fg='white',
            width=25,
            height=2,
            state=tk.DISABLED
        )
        self.start_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        # Метка для отображения информации о выбранной папке
        self.folder_info = tk.Label(
            main_frame,
            text="Папка не выбрана",
            font=("Arial", 14),
            bg='lightgray',
            fg='red',
            wraplength=700,
            justify=tk.CENTER
        )
        self.folder_info.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

        # Кнопка удаления выбора папки
        self.clear_folder_button = tk.Button(
            main_frame,
            text="Удалить выбор папки",
            command=self.clear_folder_selection,
            font=("Arial", 14),
            bg='orange',
            fg='black',
            width=20,
            height=1,
            state=tk.DISABLED
        )
        self.clear_folder_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def select_folder(self):
        """Выбор папки с фотографиями"""
        folder = filedialog.askdirectory(title="Выберите папку с фотографиями")
        if folder:
            self.photos_folder = folder
            self.update_folder_display()
            self.save_folder_selection()
            self.start_button.config(state=tk.NORMAL)
            self.clear_folder_button.config(state=tk.NORMAL)

            # Загружаем фото из выбранной папки
            self.load_photos()

            print(f"Выбрана папка: {self.photos_folder}")
            print(f"Найдено фото: {len(self.photos)}")

    def clear_folder_selection(self):
        """Удаляет выбор папки"""
        self.photos_folder = ""
        self.photos = []
        self.update_folder_display()
        self.save_folder_selection()
        self.start_button.config(state=tk.DISABLED)
        self.clear_folder_button.config(state=tk.DISABLED)

    def update_folder_display(self):
        """Обновляет отображение информации о папке"""
        if self.photos_folder:
            text = f"Выбранная папка: {self.photos_folder}\n"
            text += f"Найдено фото: {len(self.photos)}"
            self.folder_info.config(text=text, fg='green')
        else:
            self.folder_info.config(text="Папка не выбрана", fg='red')

    def save_folder_selection(self):
        """Сохраняет выбор папки в файл"""
        try:
            with open("folder_selection.txt", "w") as f:
                f.write(self.photos_folder)
        except Exception as e:
            print(f"Ошибка при сохранении выбора папки: {e}")

    def load_saved_folder(self):
        """Загружает сохраненную папку из файла"""
        try:
            if os.path.exists("folder_selection.txt"):
                with open("folder_selection.txt", "r") as f:
                    folder = f.read().strip()
                    if folder and os.path.exists(folder):
                        self.photos_folder = folder
                        self.load_photos()
                        self.update_folder_display()
                        self.start_button.config(state=tk.NORMAL)
                        self.clear_folder_button.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Ошибка при загрузке сохраненной папки: {e}")

    def load_photos(self):
        """Загружает пути к фото из указанной папки"""
        self.photos = []

        if not self.photos_folder or not os.path.exists(self.photos_folder):
            return

        print(f"Ищем фото в папке: {self.photos_folder}")

        # Поиск файлов изображений
        supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp', '*.tiff', '*.tif')

        for format in supported_formats:
            for pattern in [os.path.join(self.photos_folder, format),
                            os.path.join(self.photos_folder, format.upper())]:
                found_files = glob.glob(pattern)
                for file_path in found_files:
                    if file_path not in self.photos and os.path.isfile(file_path):
                        self.photos.append(file_path)

        # Дополнительный поиск через listdir
        for file in os.listdir(self.photos_folder):
            file_path = os.path.join(self.photos_folder, file)
            if os.path.isfile(file_path) and self.is_image_file(file_path) and file_path not in self.photos:
                self.photos.append(file_path)

        # Удаляем дубликаты и сортируем
        self.photos = list(set(self.photos))
        self.photos.sort()

        print(f"Всего найдено уникальных фото: {len(self.photos)}")

    def is_image_file(self, file_path):
        """Проверяет, является ли файл изображением"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')
        return file_path.lower().endswith(image_extensions)

    def is_file_ready(self, file_path):
        """Проверяет, доступен ли файл для чтения"""
        try:
            with open(file_path, 'rb'):
                return True
        except IOError:
            return False

    def wait_for_file_ready(self, file_path, max_attempts=15, delay=0.5):
        """Ожидает, пока файл станет доступен для чтения"""
        for attempt in range(max_attempts):
            try:
                if os.path.isfile(file_path):
                    with open(file_path, 'rb+') as f:
                        file_size = os.path.getsize(file_path)
                        if file_size > 0:
                            f.read(100)
                            f.seek(0)
                            print(f"Файл готов: {os.path.basename(file_path)} (размер: {file_size} байт)")
                            return True
            except (IOError, PermissionError, OSError) as e:
                print(f"Попытка {attempt + 1}/{max_attempts}: Файл заблокирован, повтор через {delay} сек")

            if attempt < max_attempts - 1:
                time.sleep(delay)

        print(f"Файл не стал доступен после {max_attempts} попыток: {file_path}")
        return False

    def start_viewing(self):
        """Начинает проверку фотографий"""
        if not self.photos_folder:
            messagebox.showerror("Ошибка", "Сначала выберите папку с фотографиями")
            return

        # Останавливаем мониторинг если был запущен
        self.stop_file_monitoring()

        # Сбрасываем состояния
        self.known_files = set(self.photos)
        self.analyzed_photos = {}
        self.current_photo_path = None
        self.current_photo_data = None
        self.current_photo_reference = None
        self.is_waiting_mode = False

        # Создаем интерфейс просмотра
        self.create_viewing_interface()

        # Запускаем мониторинг папки
        self.start_file_monitoring()

        # Всегда начинаем с режима ожидания
        self.show_waiting_message()

    def create_viewing_interface(self):
        """Создает интерфейс для проверки фотографий"""
        # Очищаем окно
        for widget in self.root.winfo_children():
            widget.destroy()

        self.is_waiting_mode = False

        # Фрейм для изображения
        self.image_frame = tk.Frame(self.root, bg='black')
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Метка для изображения
        self.image_label = tk.Label(self.image_frame, bg='black')
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Фрейм для информации
        self.info_frame = tk.Frame(self.root, bg='darkgray')
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Метка с информацией о фото
        self.info_label = tk.Label(
            self.info_frame,
            text="",
            font=("Arial", 18),
            bg='darkgray',
            fg='white',
            justify=tk.CENTER
        )
        self.info_label.pack(pady=10)

        # Метка для результата анализа
        self.analysis_result = tk.Label(
            self.info_frame,
            text="",
            font=("Arial", 18, "bold"),
            bg='darkgray',
            fg='white',
            justify=tk.CENTER
        )
        self.analysis_result.pack(pady=10)

        # Кнопка "Главное меню" - ОБЯЗАТЕЛЬНО СОХРАНЯЕТСЯ
        menu_button = tk.Button(
            self.info_frame,
            text="Главное меню",
            command=self.back_to_menu,
            font=("Arial", 16),
            bg='lightblue',
            fg='black',
            width=15,
            height=1
        )
        menu_button.pack(pady=10)

        # Обработка изменения размера окна
        self.root.bind("<Configure>", self.on_window_resize)

    def show_waiting_message(self):
        """Показывает сообщение об ожидании объектов анализа"""
        if hasattr(self, 'info_label') and self.info_label.winfo_exists():
            self.info_label.config(text="Ожидание объекта анализа")
            self.analysis_result.config(text="")
            self.current_photo_path = None
            self.current_photo_data = None

            # Очищаем изображение
            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                self.image_label.configure(image='')
                self.image_label.image = None

    def start_file_monitoring(self):
        """Запускает отслеживание изменений в папке"""
        if self.WATCHDOG_AVAILABLE:
            try:
                self.event_handler = PhotoWatcher(self, self.photos_folder)
                self.observer = Observer()
                self.observer.schedule(self.event_handler, self.photos_folder, recursive=False)
                self.observer.start()
                print(f"Начато отслеживание папки с помощью watchdog: {self.photos_folder}")
                return
            except Exception as e:
                print(f"Ошибка при запуске watchdog: {e}")
                self.WATCHDOG_AVAILABLE = False

        # Если watchdog недоступен, используем периодическую проверку
        print(f"Начата периодическая проверка папки: {self.photos_folder}")
        self.check_for_new_files()

    def check_for_new_files(self):
        """Периодически проверяет папку на наличие новых файлов"""
        if self.monitoring_after_id:
            self.root.after_cancel(self.monitoring_after_id)

        if not self.root.winfo_exists():
            return

        current_files = set()

        if os.path.exists(self.photos_folder):
            for file in os.listdir(self.photos_folder):
                file_path = os.path.join(self.photos_folder, file)
                if os.path.isfile(file_path) and self.is_image_file(file_path):
                    current_files.add(file_path)

        # Находим новые файлы
        new_files = current_files - self.known_files
        for file_path in new_files:
            print(f"Обнаружено новое изображение: {file_path}")
            self.add_new_photo(file_path)

        # Обновляем список известных файлов ПОСЛЕ обработки новых
        self.known_files = current_files

        # Планируем следующую проверку
        if self.root.winfo_exists():
            self.monitoring_after_id = self.root.after(2000, self.check_for_new_files)

    def stop_file_monitoring(self):
        """Останавливает отслеживание изменений в папке"""
        # Отменяем задачи мониторинга
        if self.monitoring_after_id:
            self.root.after_cancel(self.monitoring_after_id)
            self.monitoring_after_id = None

        if self.WATCHDOG_AVAILABLE and hasattr(self, 'observer'):
            try:
                self.observer.stop()
                self.observer.join()
                print("Watchdog остановлен")
            except Exception as e:
                print(f"Ошибка при остановке watchdog: {e}")

    def add_new_photo(self, photo_path):
        """Добавляет новое фото и показывает его"""
        if not photo_path or not isinstance(photo_path, str):
            print(f"Некорректный путь к файлу: {photo_path}")
            return

        # Нормализуем путь для избежания дубликатов
        photo_path = os.path.abspath(photo_path)

        # Ждем пока файл станет доступен для чтения
        if not self.wait_for_file_ready(photo_path):
            print(f"Файл не доступен для чтения: {photo_path}")
            return

        # Добавляем фото в общий список
        if photo_path not in self.photos:
            self.photos.append(photo_path)
            self.photos.sort()
            print(f"Добавлено новое фото: {os.path.basename(photo_path)}")

        # Показываем фото сразу
        self.show_photo(photo_path)

    def show_photo(self, photo_path):
        """Показывает указанное фото и не меняет его до загрузки нового"""
        if not self.root.winfo_exists():
            return

        self.current_photo_path = photo_path

        try:
            # Загружаем и масштабируем изображение
            image = Image.open(photo_path)

            # Сохраняем данные изображения в памяти
            self.current_photo_data = image

            screen_width = self.root.winfo_width()
            screen_height = self.root.winfo_height()
            img_width, img_height = image.size

            # Учитываем место для панелей управления
            control_height = 150
            max_width = screen_width - 20
            max_height = screen_height - control_height - 20

            # Масштабируем при необходимости
            if img_width > max_width or img_height > max_height:
                ratio = min(max_width / img_width, max_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Конвертируем для tkinter
            photo = ImageTk.PhotoImage(image)

            # Сохраняем ссылку на изображение, чтобы оно не удалилось сборщиком мусора
            self.current_photo_reference = photo

            # Обновляем метку с изображением
            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                self.image_label.configure(image=photo)
                self.image_label.image = photo

                # Обновляем информацию о фото
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filename = os.path.basename(photo_path)
                self.info_label.config(text=f"Фото: {filename}\nВремя загрузки: {current_time}")

                # Запускаем анализ если еще не анализировалось
                if photo_path in self.analyzed_photos:
                    result, color = self.analyzed_photos[photo_path]
                    self.show_analysis_result(result, color)
                else:
                    self.perform_analysis(photo_path)

        except Exception as e:
            print(f"Ошибка при загрузке изображения {photo_path}: {str(e)}")

    def perform_analysis(self, photo_path):
        """Выполняет анализ фото и показывает результат"""
        if not hasattr(self, 'analysis_result') or not self.analysis_result.winfo_exists():
            return

        # Проверяем, что файл все еще существует
        if not os.path.exists(photo_path):
            print(f"Файл не существует для анализа: {photo_path}")
            return

        self.analysis_result.config(text="Выполняется анализ...", fg='yellow')
        self.root.update()

        def analyze_thread():
            try:
                result, color = self.analyze_defects(photo_path)

                # Проверяем, что файл все еще существует и это текущее фото
                if (os.path.exists(photo_path) and
                        hasattr(self, 'analysis_result') and
                        self.analysis_result.winfo_exists()):

                    self.analyzed_photos[photo_path] = (result, color)

                    # Обрабатываем результат анализа
                    if result == "дефект":
                        # Переименовываем файл с дефектом
                        self.root.after(0, lambda: self.handle_defect_photo(photo_path, result, color))
                    else:
                        # Удаляем файл без дефектов
                        self.root.after(0, lambda: self.handle_good_photo(photo_path, result, color))

            except Exception as e:
                print(f"Ошибка при анализе фото: {str(e)}")
                if (hasattr(self, 'analysis_result') and
                        self.analysis_result.winfo_exists() and
                        self.current_photo_path == photo_path):
                    self.root.after(0, self.show_analysis_error)

        # Запускаем анализ в отдельном потоке
        thread = threading.Thread(target=analyze_thread, daemon=True)
        thread.start()

    def handle_defect_photo(self, photo_path, result, color):
        """Обрабатывает фото с дефектом - переименовывает в той же папке"""
        # Показываем результат анализа
        if self.current_photo_path == photo_path:
            self.show_analysis_result(result, color)

        # Переименовываем файл с дефектом
        success = self.rename_defect_file(photo_path)

        if success:
            print(f"Файл с дефектом переименован: {os.path.basename(photo_path)}")
        else:
            print(f"Ошибка при переименовании файла с дефектом: {photo_path}")

    def handle_good_photo(self, photo_path, result, color):
        """Обрабатывает хорошее фото - удаляет файл"""
        # Показываем результат анализа
        if self.current_photo_path == photo_path:
            self.show_analysis_result(result, color)

        # Удаляем файл без дефектов
        success = self.delete_good_file(photo_path)

        if success:
            print(f"Хороший файл удален: {os.path.basename(photo_path)}")
            # Если удален текущий файл, показываем сообщение ожидания
            if self.current_photo_path == photo_path:
                self.current_photo_path = None
                self.show_waiting_message()
        else:
            print(f"Ошибка при удалении хорошего файла: {photo_path}")

    def rename_defect_file(self, file_path):
        """Переименовывает файл с дефектом в текущей папке"""
        max_attempts = 5
        delay_between_attempts = 1  # секунда

        for attempt in range(max_attempts):
            try:
                if not os.path.isfile(file_path):
                    print(f"Попытка {attempt + 1}/{max_attempts}: Файл не существует: {file_path}")
                    return False

                # Получаем расширение файла
                file_ext = os.path.splitext(file_path)[1]

                # Генерируем новое имя в формате: Год-месяц-день часы-минуты-секунды
                now = datetime.now()
                new_name = now.strftime("%Y-%m-%d %H-%M-%S") + file_ext

                # Полный путь к новому файлу в той же папке
                new_path = os.path.join(os.path.dirname(file_path), new_name)

                # Если файл с таким именем уже существует, добавляем номер
                counter = 1
                while os.path.exists(new_path):
                    new_name = now.strftime("%Y-%m-%d %H-%M-%S") + f"_{counter}" + file_ext
                    new_path = os.path.join(os.path.dirname(file_path), new_name)
                    counter += 1

                # Переименовываем файл
                os.rename(file_path, new_path)
                print(f"Файл переименован: {os.path.basename(file_path)} -> {new_name}")

                # Обновляем пути в списках
                if file_path in self.photos:
                    self.photos.remove(file_path)
                    self.photos.append(new_path)
                    self.photos.sort()

                if file_path in self.analyzed_photos:
                    self.analyzed_photos[new_path] = self.analyzed_photos.pop(file_path)

                if file_path == self.current_photo_path:
                    self.current_photo_path = new_path

                return True

            except PermissionError as e:
                print(
                    f"Попытка {attempt + 1}/{max_attempts}: Файл заблокирован, повтор через {delay_between_attempts} сек: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay_between_attempts)
                else:
                    print(f"Не удалось переименовать файл после {max_attempts} попыток: {file_path}")
                    return False

            except Exception as e:
                print(f"Попытка {attempt + 1}/{max_attempts}: Ошибка при переименовании файла: {e}")
                return False

        return False

    def delete_good_file(self, file_path):
        """Удаляет хороший файл"""
        max_attempts = 5
        delay_between_attempts = 1  # секунда

        for attempt in range(max_attempts):
            try:
                if not os.path.isfile(file_path):
                    print(f"Попытка {attempt + 1}/{max_attempts}: Файл не существует: {file_path}")
                    return False

                # Удаляем файл
                os.remove(file_path)
                print(f"Файл удален: {os.path.basename(file_path)}")

                # Обновляем пути в списках
                if file_path in self.photos:
                    self.photos.remove(file_path)

                if file_path in self.analyzed_photos:
                    del self.analyzed_photos[file_path]

                return True

            except PermissionError as e:
                print(
                    f"Попытка {attempt + 1}/{max_attempts}: Файл заблокирован, повтор через {delay_between_attempts} сек: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay_between_attempts)
                else:
                    print(f"Не удалось удалить файл после {max_attempts} попыток: {file_path}")
                    return False

            except Exception as e:
                print(f"Попытка {attempt + 1}/{max_attempts}: Ошибка при удалении файла: {e}")
                return False

        return False

    def analyze_defects(self, photo_path):
        """
        Анализирует фото на наличие дефектов с помощью нейронной сети
        """
        # Если модель не загружена, используем демо-режим
        if self.model is None:
            print("❌ Модель не загружена, используется демо-режим")
            return self.analyze_defects_demo(photo_path)

        try:
            # Проверяем существование файла
            if not os.path.exists(photo_path):
                print(f"❌ Файл {photo_path} не найден")
                return "ошибка", 'red'

            # Загрузка и обработка изображения
            img = Image.open(photo_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Изменение размера и нормализация
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Предсказание
            prediction = self.model.predict(img_array, verbose=0)
            defect_prob = float(prediction[0][0])

            # ИНВЕРТИРОВАННАЯ ЛОГИКА: если нейросеть говорит "дефект", мы считаем "не дефект" и наоборот
            if defect_prob >= 0.5:
                # Нейросеть считает, что это дефект -> мы считаем, что это НЕ дефект
                result = "не дефект"
                color = 'green'
            else:
                # Нейросеть считает, что это не дефект -> мы считаем, что это ДЕФЕКТ
                result = "дефект"
                color = 'red'

            print(f"🔍 Анализ {os.path.basename(photo_path)}: {result} (вероятность нейросети: {defect_prob:.3f})")
            return result, color

        except Exception as e:
            print(f"❌ Ошибка анализа {photo_path}: {e}")
            return "ошибка", 'red'

    def analyze_defects_demo(self, photo_path):
        """Демо-режим анализа фото на дефекты (используется если модель недоступна)"""
        try:
            # Детерминированный результат на основе имени файла
            filename = os.path.basename(photo_path)
            file_hash = hash(filename) % 2  # 0 или 1

            # ИНВЕРТИРОВАННАЯ ЛОГИКА для демо-режима
            if file_hash == 0:
                result = "не дефект"
                color = 'green'
            else:
                result = "дефект"
                color = 'red'

            # Имитация времени анализа
            time.sleep(1)

            print(f"🔍 Демо-анализ {filename}: {result}")
            return result, color

        except Exception as e:
            print(f"❌ Ошибка демо-анализа {photo_path}: {e}")
            return "ошибка", 'red'

    def show_analysis_result(self, result, color):
        """Показывает результат анализа"""
        if hasattr(self, 'analysis_result') and self.analysis_result.winfo_exists():
            self.analysis_result.config(text=f"Результат анализа: {result}", fg=color, bg='darkgray')

    def show_analysis_error(self):
        """Показывает ошибку анализа"""
        if hasattr(self, 'analysis_result') and self.analysis_result.winfo_exists():
            self.analysis_result.config(text="Ошибка анализа", fg='red', bg='darkgray')

    def on_window_resize(self, event):
        """Обрабатывает изменение размера окна и перерисовывает изображение"""
        if (hasattr(self, 'current_photo_path') and self.current_photo_path and
                self.root.winfo_exists() and hasattr(self, 'current_photo_data')):
            # Перерисовываем изображение из данных в памяти
            self.root.after(100, lambda: self._redisplay_current_photo())

    def _redisplay_current_photo(self):
        """Перерисовывает текущее фото из данных в памяти"""
        if not hasattr(self, 'current_photo_data') or not self.current_photo_data:
            return

        try:
            image = self.current_photo_data
            screen_width = self.root.winfo_width()
            screen_height = self.root.winfo_height()
            img_width, img_height = image.size

            # Учитываем место для панелей управления
            control_height = 150
            max_width = screen_width - 20
            max_height = screen_height - control_height - 20

            # Масштабируем при необходимости
            if img_width > max_width or img_height > max_height:
                ratio = min(max_width / img_width, max_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Конвертируем для tkinter
            photo = ImageTk.PhotoImage(image)

            # Сохраняем ссылку на изображение
            self.current_photo_reference = photo

            # Обновляем метку с изображением
            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                self.image_label.configure(image=photo)
                self.image_label.image = photo

        except Exception as e:
            print(f"Ошибка при перерисовке изображения: {str(e)}")

    def back_to_menu(self):
        """Возврат в главное меню"""
        print("Возврат в главное меню")

        # Останавливаем все процессы
        self.stop_file_monitoring()

        # Отвязываем обработчики
        self.root.unbind("<Configure>")

        # Сбрасываем состояние
        self.current_photo_path = None
        self.current_photo_data = None
        self.current_photo_reference = None
        self.is_waiting_mode = False

        # Возвращаемся в главное меню
        self.create_main_menu()
        self.load_saved_folder()

    def toggle_fullscreen(self, event=None):
        """Переключает режим полного экрана"""
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            # Используем state('zoomed') для правильного определения рабочей области
            try:
                self.root.state('zoomed')
            except:
                # Альтернативный метод
                work_width, work_height = self.get_work_area()
                self.root.geometry(f"{work_width}x{work_height}+0+0")
        else:
            # Возвращаем нормальный размер
            self.root.state('normal')
            self.root.geometry("800x600+100+100")

        if hasattr(self, 'current_photo_path') and self.current_photo_path and self.root.winfo_exists():
            self.root.after(100, self._redisplay_current_photo)


def main():
    root = tk.Tk()
    app = PhotoViewer(root)

    # Обработка клавиши Escape для выхода из полноэкранного режима
    root.bind("<Escape>", app.toggle_fullscreen)

    def on_closing():
        app.stop_file_monitoring()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
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
        self.handle_event(event)

    def on_moved(self, event):
        self.handle_event(event)

    def on_modified(self, event):
        self.handle_event(event)

    def handle_event(self, event):
        src_path = getattr(event, 'src_path', None)
        dest_path = getattr(event, 'dest_path', None)
        file_path = dest_path if dest_path else src_path

        if not event.is_directory and self.is_image_file(file_path):
            print(f"Обнаружено изменение: {file_path}")
            self.app.root.after(1000, lambda: self.app.add_new_photo(file_path))

    def is_image_file(self, file_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')
        return file_path.lower().endswith(image_extensions)


class PhotoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Проверка на дефекты")
        self.is_fullscreen = True

        # Устанавливаем полноэкранный режим
        try:
            self.root.state('zoomed')
        except:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        # Инициализация переменных
        self.photos_folder = ""
        self.WATCHDOG_AVAILABLE = WATCHDOG_AVAILABLE
        self.model = None
        self.photos = []
        self.current_photo_path = None
        self.current_photo_data = None
        self.known_files = set()
        self.analyzed_photos = {}
        self.current_photo_reference = None
        self.is_waiting_mode = False
        self.monitoring_after_id = None

        # Загрузка модели
        if TENSORFLOW_AVAILABLE:
            try:
                self.model = load_model('defect_detection_continued.h5')
                print("✅ Модель нейронной сети загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
                self.model = None

        self.create_main_menu()
        self.load_saved_folder()

    def create_main_menu(self):
        """Создает главное меню с выбором папки"""
        for widget in self.root.winfo_children():
            widget.destroy()

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
            text = f"Выбранная папка: {self.photos_folder}\nНайдено фото: {len(self.photos)}"
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
        supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp', '*.tiff', '*.tif')

        for format in supported_formats:
            for pattern in [os.path.join(self.photos_folder, format),
                            os.path.join(self.photos_folder, format.upper())]:
                found_files = glob.glob(pattern)
                for file_path in found_files:
                    if file_path not in self.photos and os.path.isfile(file_path):
                        self.photos.append(file_path)

        for file in os.listdir(self.photos_folder):
            file_path = os.path.join(self.photos_folder, file)
            if os.path.isfile(file_path) and self.is_image_file(file_path) and file_path not in self.photos:
                self.photos.append(file_path)

        self.photos = list(set(self.photos))
        self.photos.sort()
        print(f"Всего найдено уникальных фото: {len(self.photos)}")

    def is_image_file(self, file_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')
        return file_path.lower().endswith(image_extensions)

    def is_file_ready(self, file_path):
        try:
            with open(file_path, 'rb'):
                return True
        except IOError:
            return False

    def wait_for_file_ready(self, file_path, max_attempts=15, delay=0.5):
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

        self.stop_file_monitoring()
        self.known_files = set(self.photos)
        self.analyzed_photos = {}
        self.current_photo_path = None
        self.current_photo_data = None
        self.current_photo_reference = None
        self.is_waiting_mode = False

        self.create_viewing_interface()
        self.start_file_monitoring()
        self.show_waiting_message()

    def create_viewing_interface(self):
        """Создает интерфейс для проверки фотографий с кнопкой в правом нижнем углу"""
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

        # ⭐ ОДНА кнопка "Главное меню" в правом нижнем углу - в два раза больше
        self.menu_button = tk.Button(
            self.root,  # Размещаем непосредственно в корневом окне
            text="Главное меню",
            command=self.back_to_menu,
            font=("Arial", 20, "bold"),  # Увеличенный шрифт
            bg='lightblue',
            fg='black',
            width=30,  # В два раза больше по ширине (было 15)
            height=3,  # В два раза больше по высоте (было 1-2)
            borderwidth=3,
            relief="raised"
        )
        # Размещение в правом нижнем углу :cite[1]:cite[2]
        self.menu_button.place(relx=1.0, rely=1.0, anchor='se', x=-20, y=-20)

        self.root.bind("<Configure>", self.on_window_resize)

    def show_waiting_message(self):
        """Показывает сообщение об ожидании объектов анализа"""
        if hasattr(self, 'info_label') and self.info_label.winfo_exists():
            self.info_label.config(text="Ожидание объекта анализа")
            self.analysis_result.config(text="")
            self.current_photo_path = None
            self.current_photo_data = None

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

        new_files = current_files - self.known_files
        for file_path in new_files:
            print(f"Обнаружено новое изображение: {file_path}")
            self.add_new_photo(file_path)

        self.known_files = current_files

        if self.root.winfo_exists():
            self.monitoring_after_id = self.root.after(2000, self.check_for_new_files)

    def stop_file_monitoring(self):
        """Останавливает отслеживание изменений в папке"""
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
            return

        photo_path = os.path.abspath(photo_path)
        if not self.wait_for_file_ready(photo_path):
            return

        if photo_path not in self.photos:
            self.photos.append(photo_path)
            self.photos.sort()
            print(f"Добавлено новое фото: {os.path.basename(photo_path)}")

        self.show_photo(photo_path)

    def show_photo(self, photo_path):
        """Показывает указанное фото"""
        if not self.root.winfo_exists():
            return

        self.current_photo_path = photo_path

        try:
            image = Image.open(photo_path)
            self.current_photo_data = image

            screen_width = self.root.winfo_width()
            screen_height = self.root.winfo_height()
            img_width, img_height = image.size

            control_height = 150
            max_width = screen_width - 20
            max_height = screen_height - control_height - 20

            if img_width > max_width or img_height > max_height:
                ratio = min(max_width / img_width, max_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.current_photo_reference = photo

            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                self.image_label.configure(image=photo)
                self.image_label.image = photo

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filename = os.path.basename(photo_path)
                self.info_label.config(text=f"Фото: {filename}\nВремя загрузки: {current_time}")

                if photo_path in self.analyzed_photos:
                    result, color = self.analyzed_photos[photo_path]
                    self.show_analysis_result(result, color)
                else:
                    self.perform_analysis(photo_path)

        except Exception as e:
            print(f"Ошибка при загрузке изображения {photo_path}: {str(e)}")

    def perform_analysis(self, photo_path):
        """Выполняет анализ фото в отдельном потоке"""
        if not hasattr(self, 'analysis_result') or not self.analysis_result.winfo_exists():
            return

        if not os.path.exists(photo_path):
            return

        self.analysis_result.config(text="Выполняется анализ...", fg='yellow')
        self.root.update_idletasks()

        def analyze_in_thread():
            try:
                result, color = self.analyze_defects(photo_path)
                self.root.after(0, lambda: self.finish_analysis(photo_path, result, color))
            except Exception as e:
                print(f"Ошибка при анализе фото: {str(e)}")
                self.root.after(0, self.show_analysis_error)

        thread = threading.Thread(target=analyze_in_thread, daemon=True)
        thread.start()

    def finish_analysis(self, photo_path, result, color):
        """Завершает анализ в главном потоке"""
        try:
            if not self.root.winfo_exists():
                return

            self.analyzed_photos[photo_path] = (result, color)
            self.show_analysis_result(result, color)

            if result == "дефект":
                self.handle_defect_photo(photo_path, result, color)
            else:
                self.handle_good_photo(photo_path, result, color)

        except Exception as e:
            print(f"Ошибка при завершении анализа: {e}")
            self.show_analysis_error()

    def handle_defect_photo(self, photo_path, result, color):
        """Обрабатывает фото с дефектом"""
        try:
            if (hasattr(self, 'analysis_result') and
                    self.analysis_result.winfo_exists() and
                    self.current_photo_path == photo_path):
                self.show_analysis_result(result, color)

            def rename_thread():
                success = self.rename_defect_file(photo_path)
                if success:
                    print(f"Файл с дефектом переименован: {os.path.basename(photo_path)}")
                else:
                    print(f"Ошибка при переименовании файла с дефектом: {photo_path}")

            thread = threading.Thread(target=rename_thread, daemon=True)
            thread.start()

        except Exception as e:
            print(f"Ошибка при обработке дефекта: {e}")

    def handle_good_photo(self, photo_path, result, color):
        """Обрабатывает хорошее фото - удаляет файл"""
        if self.current_photo_path == photo_path:
            self.show_analysis_result(result, color)

        success = self.delete_good_file(photo_path)

        if success:
            print(f"Хороший файл удален: {os.path.basename(photo_path)}")
            if self.current_photo_path == photo_path:
                self.current_photo_path = None
                self.show_waiting_message()
        else:
            print(f"Ошибка при удалении хорошего файла: {photo_path}")

    def rename_defect_file(self, file_path):
        """Переименовывает файл с дефектом в текущей папке"""
        max_attempts = 5
        delay_between_attempts = 1

        for attempt in range(max_attempts):
            try:
                if not os.path.isfile(file_path):
                    return False

                file_ext = os.path.splitext(file_path)[1]
                now = datetime.now()
                new_name = now.strftime("%Y-%m-%d %H-%M-%S") + file_ext
                new_path = os.path.join(os.path.dirname(file_path), new_name)

                counter = 1
                while os.path.exists(new_path):
                    new_name = now.strftime("%Y-%m-%d %H-%M-%S") + f"_{counter}" + file_ext
                    new_path = os.path.join(os.path.dirname(file_path), new_name)
                    counter += 1

                os.rename(file_path, new_path)
                print(f"Файл переименован: {os.path.basename(file_path)} -> {new_name}")

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
                if attempt < max_attempts - 1:
                    time.sleep(delay_between_attempts)
                else:
                    return False
            except Exception as e:
                return False

        return False

    def delete_good_file(self, file_path):
        """Удаляет хороший файл"""
        max_attempts = 5
        delay_between_attempts = 1

        for attempt in range(max_attempts):
            try:
                if not os.path.isfile(file_path):
                    return False

                os.remove(file_path)
                print(f"Файл удален: {os.path.basename(file_path)}")

                if file_path in self.photos:
                    self.photos.remove(file_path)

                if file_path in self.analyzed_photos:
                    del self.analyzed_photos[file_path]

                return True

            except PermissionError as e:
                if attempt < max_attempts - 1:
                    time.sleep(delay_between_attempts)
                else:
                    return False
            except Exception as e:
                return False

        return False

    def analyze_defects(self, photo_path):
        """Анализирует фото на наличие дефектов с помощью нейронной сети"""
        if self.model is None:
            return self.analyze_defects_demo(photo_path)

        try:
            if not os.path.exists(photo_path):
                return "ошибка", 'red'

            img = Image.open(photo_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array, verbose=0)
            defect_prob = float(prediction[0][0])

            if defect_prob >= 0.5:
                result = "не дефект"
                color = 'green'
            else:
                result = "дефект"
                color = 'red'

            print(f"🔍 Анализ {os.path.basename(photo_path)}: {result} (вероятность: {defect_prob:.3f})")
            return result, color

        except Exception as e:
            print(f"❌ Ошибка анализа {photo_path}: {e}")
            return "ошибка", 'red'

    def analyze_defects_demo(self, photo_path):
        """Демо-режим анализа фото на дефекты"""
        try:
            filename = os.path.basename(photo_path)
            file_hash = hash(filename) % 2

            if file_hash == 0:
                result = "не дефект"
                color = 'green'
            else:
                result = "дефект"
                color = 'red'

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
        """Обрабатывает изменение размера окна"""
        if (hasattr(self, 'current_photo_path') and self.current_photo_path and
                self.root.winfo_exists() and hasattr(self, 'current_photo_data')):
            self.root.after(100, self._redisplay_current_photo)

    def _redisplay_current_photo(self):
        """Перерисовывает текущее фото"""
        if not hasattr(self, 'current_photo_data') or not self.current_photo_data:
            return

        try:
            image = self.current_photo_data
            screen_width = self.root.winfo_width()
            screen_height = self.root.winfo_height()
            img_width, img_height = image.size

            control_height = 150
            max_width = screen_width - 20
            max_height = screen_height - control_height - 20

            if img_width > max_width or img_height > max_height:
                ratio = min(max_width / img_width, max_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.current_photo_reference = photo

            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                self.image_label.configure(image=photo)
                self.image_label.image = photo

        except Exception as e:
            print(f"Ошибка при перерисовке изображения: {str(e)}")

    def back_to_menu(self):
        """Возврат в главное меню - кнопка пропадает только здесь"""
        print("Возврат в главное меню...")

        try:
            self.stop_file_monitoring()
            self.root.unbind("<Configure>")

            self.current_photo_path = None
            self.current_photo_data = None
            self.current_photo_reference = None
            self.is_waiting_mode = False

            # ⭐ Кнопка автоматически уничтожится при очистке окна в create_main_menu
            self.create_main_menu()
            self.load_saved_folder()

            print("Успешный возврат в главное меню")

        except Exception as e:
            print(f"Ошибка при возврате в меню: {e}")
            try:
                self.create_main_menu()
            except:
                messagebox.showerror("Ошибка", "Не удалось вернуться в меню")

    def toggle_fullscreen(self, event=None):
        """Переключает режим полного экрана"""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            try:
                self.root.state('zoomed')
            except:
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        else:
            self.root.state('normal')
            self.root.geometry("800x600+100+100")

        if hasattr(self, 'current_photo_path') and self.current_photo_path and self.root.winfo_exists():
            self.root.after(100, self._redisplay_current_photo)


def main():
    root = tk.Tk()
    app = PhotoViewer(root)
    root.bind("<Escape>", app.toggle_fullscreen)

    def on_closing():
        app.stop_file_monitoring()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

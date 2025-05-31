import customtkinter
import sqlite3
from tkinter import filedialog, TclError
from Analizer import Main_Predict, add_to_model


class Table(customtkinter.CTkScrollableFrame):
    def __init__(self, parent, headers, data, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        for col, text in enumerate(headers):
            label = customtkinter.CTkLabel(
                self, text=text, fg_color="gray", corner_radius=5)
            label.grid(row=0, column=col, padx=5, pady=5, sticky="ew")

        for row, row_data in enumerate(data, start=1):
            for col, cell in enumerate(row_data):
                label = customtkinter.CTkLabel(self, text=str(cell))
                label.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

    def update_data(self, headers, new_data):
        for widget in self.winfo_children():
            widget.destroy()
        self.__init__(self.master, headers, new_data)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        customtkinter.set_default_color_theme("theme.json")

        self.title("Graphology")
        self.geometry("900x500")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=9)
        self.grid_rowconfigure(0, weight=10)

        self.frame_stack = []
        self.class_list = []
        self.logic_features = []
        self.range_features = []
        self.scalar_features = []
        self.fillness_flag = False

        self.button_frame = customtkinter.CTkFrame(self)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(6, weight=1)
        self.button_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.active_frame = customtkinter.CTkFrame(self)
        self.active_frame.grid_columnconfigure(0, weight=1)
        self.active_frame.grid_rowconfigure(1, weight=1)
        self.active_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.btn_edit_db = customtkinter.CTkButton(
            self.button_frame,
            text="Редактировать базу знаний",
            command=self.Edit_DataKnowledge_Pressed
        )
        self.btn_edit_db.grid(
            row=0,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        self.btn_solve_task = customtkinter.CTkButton(
            self.button_frame,
            text="Решить задачу(выбор)",
            command=self.Solve_Task
        )
        self.btn_solve_task.grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        self.btn_solve_task_neuro = customtkinter.CTkButton(
            self.button_frame,
            text="Решить задачу(нейро)",
            command=self.Solve_Task_neuro
        )
        self.btn_solve_task_neuro.grid(
            row=2,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        self.default_buttons = [self.btn_edit_db,
                                self.btn_solve_task, self.btn_solve_task_neuro]
        self.current_buttons = list(self.default_buttons)

        self.frame_stack.append(self.default_buttons.copy())

    def Solve_Task_neuro(self):
        self.check_fillness()

        if self.fillness_flag:

            self.frame_stack.append(self.current_buttons.copy())

            self.main_frame = customtkinter.CTkScrollableFrame(
                self.active_frame)
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(8, weight=1)
            self.main_frame.grid(row=1, column=0, padx=5,
                                 pady=5, sticky="nsew")

            self.back_frame = customtkinter.CTkFrame(self.active_frame)
            self.back_frame.grid_columnconfigure(4, weight=1)
            self.back_frame.grid_rowconfigure(0, weight=1)
            self.back_frame.grid(row=0, column=0, padx=5,
                                 pady=5, sticky="nsew")

            self.btn_back = customtkinter.CTkButton(
                self.back_frame, text="Назад",
                command=self.show_previous_buttons)
            self.btn_back.grid(row=2, column=0, padx=10, pady=10, sticky="w")

            self.image_path = customtkinter.StringVar(
                value="Изображение не выбрано")

            self.btn_load_image = customtkinter.CTkButton(
                self.main_frame, text="Загрузить изображение",
                command=self.load_image)
            self.btn_load_image.grid(
                row=0, column=0, padx=10, pady=5, sticky="nw")

            self.lbl_image_path = customtkinter.CTkLabel(
                self.main_frame, textvariable=self.image_path)
            self.lbl_image_path.grid(
                row=1, column=0, padx=10, pady=5, sticky="nwe")

            self.btn_predict = customtkinter.CTkButton(
                self.main_frame, text="Предсказать класс",
                command=self.predict_class)
            self.btn_predict.grid(
                row=0, column=0, padx=10, pady=5, sticky="ne")

            if hasattr(self, 'btn_submit') and self.btn_submit.winfo_exists():
                self.btn_submit.destroy()
            self.current_buttons = self.default_buttons.copy()

        else:
            self.show_message(text="полнота не пройдена")

    def Edit_DataKnowledge_Pressed(self):

        self.frame_stack.append(self.current_buttons.copy())

        self.Hide_current_elements()
        self.clear_active_frame()

        self.btn_classes = customtkinter.CTkButton(
            self.button_frame, text="Классы", command=self.show_classes)
        self.btn_classes.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.btn_features = customtkinter.CTkButton(
            self.button_frame, text="Признаки", command=self.show_features)
        self.btn_features.grid(row=1, column=0, padx=10,
                               pady=10, sticky="nsew")

        self.btn_values = customtkinter.CTkButton(
            self.button_frame, text="Значения признаков",
            command=self.show_feature_values)
        self.btn_values.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.btn_class_features = customtkinter.CTkButton(
            # show_class_features
            self.button_frame, text="Признаки класса",
            command=self.show_class_features)
        self.btn_class_features.grid(
            row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.btn_feature_values = customtkinter.CTkButton(
            # show_class_feature_values
            self.button_frame, text="Значения признаков классов",
            command=self.show_class_feature_values)
        self.btn_feature_values.grid(
            row=4, column=0, padx=10, pady=10, sticky="nsew")

        self.btn_fillness = customtkinter.CTkButton(
            self.button_frame, text="Проверить полноту",
            command=self.check_fillness)
        self.btn_fillness.grid(row=5, column=0, padx=10,
                               pady=10, sticky="nsew")

        self.main_frame = customtkinter.CTkFrame(self.active_frame)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.back_frame = customtkinter.CTkFrame(self.active_frame)
        self.back_frame.grid_columnconfigure(4, weight=1)
        self.back_frame.grid_rowconfigure(0, weight=1)
        self.back_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.btn_back = customtkinter.CTkButton(
            self.back_frame, text="Назад", command=self.show_previous_buttons)
        self.btn_back.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.current_buttons = [self.btn_classes, self.btn_features,
                                self.btn_values,
                                self.btn_class_features,
                                self.btn_feature_values,
                                self.btn_fillness,
                                self.btn_back]

    def show_classes(self):
        self.frame_stack.append(self.current_buttons.copy())
        self.clear_main_frame()
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.entry_class = customtkinter.CTkEntry(self.main_frame)
        self.entry_class.grid(row=0, column=0, padx=10, pady=5, sticky="we")
        self.btn_add_class = customtkinter.CTkButton(
            self.main_frame, text="Добавить", command=self.add_class)
        self.btn_add_class.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        self.label_classes = customtkinter.CTkLabel(
            self.main_frame, text="Список классов")
        self.label_classes.grid(
            row=1, column=0, columnspan=2, padx=10, pady=5, sticky="wn")
        self.class_listbox = customtkinter.CTkTextbox(
            self.main_frame, height=200, width=300, state="normal")
        self.class_listbox.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.class_list = [class_name[0]
                           for class_name in self.get_class_names()]
        self.class_listbox.delete("0.0", "end")
        for class_name in self.class_list:
            self.class_listbox.insert("end", class_name + "\n")
        self.class_listbox.configure(state="disabled")
        self.update_class_listbox()
        self.btn_delete_class = customtkinter.CTkButton(
            self.main_frame, text="Удалить", command=self.delete_class)
        self.btn_delete_class.grid(
            row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.current_buttons = [self.btn_back]

    def show_features(self):
        self.frame_stack.append(self.current_buttons.copy())
        self.clear_main_frame()
        self.btn_logic_features = customtkinter.CTkButton(
            self.back_frame, text="Логические признаки",
            command=self.show_logic_features)
        self.btn_logic_features.grid(
            row=2, column=1, padx=10, pady=5, sticky="nsew")
        self.btn_size_features = customtkinter.CTkButton(
            self.back_frame, text="Размерные признаки",
            command=self.show_range_features)
        self.btn_size_features.grid(
            row=2, column=2, padx=10, pady=5, sticky="nsew")
        self.btn_scalar_features = customtkinter.CTkButton(
            self.back_frame, text="Скалярные признаки",
            command=self.show_scalar_features)
        self.btn_scalar_features.grid(
            row=2, column=3, padx=10, pady=5, sticky="nsew")
        self.current_buttons = [self.btn_logic_features,
                                self.btn_size_features,
                                self.btn_scalar_features,
                                self.btn_back]

    def show_logic_features(self):
        self.clear_main_frame()

        self.entry_logic_feature = customtkinter.CTkEntry(self.main_frame)
        self.entry_logic_feature.grid(
            row=0, column=0, padx=10, pady=5, sticky="new")

        self.btn_add_logic_feature = customtkinter.CTkButton(
            self.main_frame, text="Добавить признак",
            command=self.add_logic_feature)
        self.btn_add_logic_feature.grid(
            row=0, column=1, padx=20, pady=5, sticky="new")

        self.label_logic_features = customtkinter.CTkLabel(
            self.main_frame, text="Логические признаки")
        self.label_logic_features.grid(
            row=1, column=0, padx=10, pady=5, sticky="ew")

        self.logic_feature_listbox = customtkinter.CTkTextbox(
            self.main_frame, height=200, width=300, state="normal")
        self.logic_feature_listbox.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.logic_features = [logic_name[0]
                               for logic_name in self.get_logic_names()]
        self.logic_feature_listbox.delete("0.0", "end")
        for logic_name in self.logic_features:
            self.logic_feature_listbox.insert("end", logic_name + "\n")
        self.logic_feature_listbox.configure(state="disabled")
        self.update_logic_feature_listbox()

        self.btn_delete_logic_feature = customtkinter.CTkButton(
            self.main_frame, text="Удалить", command=self.delete_logic_feature)
        self.btn_delete_logic_feature.grid(
            row=3, column=0, padx=10, pady=5, sticky="nsew")

    def show_range_features(self):
        self.clear_main_frame()

        self.entry_range_feature = customtkinter.CTkEntry(self.main_frame)
        self.entry_range_feature.grid(
            row=0, column=0, padx=10, pady=5, sticky="new")

        self.btn_add_range_feature = customtkinter.CTkButton(
            self.main_frame, text="Добавить признак",
            command=self.add_range_feature)
        self.btn_add_range_feature.grid(
            row=0, column=1, padx=10, pady=5, sticky="new")

        self.label_range_features = customtkinter.CTkLabel(
            self.main_frame, text="Размерные признаки")
        self.label_range_features.grid(
            row=1, column=0, padx=10, pady=5, sticky="ew")

        self.range_feature_listbox = customtkinter.CTkTextbox(
            self.main_frame, height=200, width=300, state="normal")
        self.range_feature_listbox.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.range_features = [row[0] for row in self.get_range_names()]
        self.range_feature_listbox.delete("0.0", "end")

        for range_name in self.range_features:
            self.range_feature_listbox.insert("end", range_name + "\n")
        self.range_feature_listbox.configure(state="disabled")
        self.update_range_feature_listbox()

        self.btn_delete_range_feature = customtkinter.CTkButton(
            self.main_frame, text="Удалить", command=self.delete_range_feature)
        self.btn_delete_range_feature.grid(
            row=3, column=0, padx=10, pady=5, sticky="ew")

    def show_scalar_features(self):
        self.clear_main_frame()

        self.entry_scalar_feature = customtkinter.CTkEntry(self.main_frame)
        self.entry_scalar_feature.grid(
            row=0, column=0, padx=10, pady=5, sticky="new")

        self.btn_add_scalar_feature = customtkinter.CTkButton(
            self.main_frame, text="Добавить признак",
            command=self.add_scalar_feature)
        self.btn_add_scalar_feature.grid(
            row=0, column=1, padx=10, pady=5, sticky="new")

        self.label_scalar_features = customtkinter.CTkLabel(
            self.main_frame, text="Скалярные признаки")
        self.label_scalar_features.grid(
            row=1, column=0, padx=10, pady=5, sticky="ew")

        self.scalar_feature_listbox = customtkinter.CTkTextbox(
            self.main_frame, height=200, width=300, state="normal")
        self.scalar_feature_listbox.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.scalar_features = list({row[0]
                                    for row in self.get_scalar_names()})
        self.scalar_feature_listbox.delete("0.0", "end")
        for scalar_name in self.scalar_features:
            self.scalar_feature_listbox.insert("end", scalar_name + "\n")
        self.scalar_feature_listbox.configure(state="disabled")
        self.update_scalar_feature_listbox()

        self.btn_delete_scalar_feature = customtkinter.CTkButton(
            self.main_frame, text="Удалить",
            command=self.delete_scalar_feature)
        self.btn_delete_scalar_feature.grid(
            row=3, column=0, padx=10, pady=5, sticky="ew")

    def show_class_features(self):
        self.frame_stack.append(self.current_buttons.copy())
        class_names = [row[0] for row in self.SQL_Lite("get_classes")]

        self.class_name = customtkinter.CTkComboBox(
            self.main_frame, values=class_names)
        self.class_name.grid(row=0, column=0, padx=10, pady=5, sticky="wne")

        feature_names = list({row[0] for row in self.get_feature_names()})
        # , command=lambda event: update_feature_values(event)
        self.feature_name = customtkinter.CTkComboBox(
            self.main_frame, values=feature_names)
        self.feature_name.grid(row=0, column=1, padx=10, pady=5, sticky="wne")

        self.btn_add_class = customtkinter.CTkButton(
            self.main_frame, text="Добавить", command=self.add_class_features)
        self.btn_add_class.grid(row=0, column=2, padx=10, pady=5, sticky="ne")

        self.btn_delete_class_feature = customtkinter.CTkButton(
            self.main_frame, text="удалить",
            command=self.remove_class_features)
        self.btn_delete_class_feature.grid(
            row=0, column=3, padx=10, pady=5, sticky="ne")

        data = self.SQL_Lite("get_data")
        headers = ["Class Name", "Feature"]

        self.table_class_features = Table(
            self.main_frame, headers, data, height=300)
        self.table_class_features.grid(
            row=2, column=0, padx=10, pady=10, sticky="new", columnspan=3)
        self.update_table(data, headers)

    def remove_class_features(self):

        class_name = self.class_name.get()
        feature_name = self.feature_name.get()

        feature_type = ""
        if feature_name in [feature[0] for feature in
                            self.SQL_Lite("get_logic_features")]:
            feature_type = "logic"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_range_features")]:
            feature_type = "range"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_scalar_features")]:
            feature_type = "scalar"

        table_map = {
            "logic": "Class_Logic_Feature",
            "range": "Class_Range_Feature",
            "scalar": "Class_Scalar_Feature",
        }
        table_name = table_map.get(feature_type)

        if not table_name:

            return

        self.SQL_Lite("rem_Feature_from_"+str(table_name),
                      name=class_name, value1=feature_name)

    def delete_selected_feature(self):
        class_name = self.class_name.get()
        feature_name = self.feature_name.get()

        feature_type = ""
        if feature_name in [feature[0] for feature in
                            self.SQL_Lite("get_logic_features")]:
            feature_type = "logic"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_range_features")]:
            feature_type = "range"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_scalar_features")]:
            feature_type = "scalar"

        table_map = {
            "logic": "Class_Logic_Feature",
            "range": "Class_Range_Feature",
            "scalar": "Class_Scalar_Feature",
        }
        table_name = table_map.get(feature_type)

        if not table_name:

            return

        if (class_name, feature_name) not in self.SQL_Lite("get_data_class_feature"):
            self.SQL_Lite("remove_from"+str(table_name),
                          name=class_name, value1=feature_name)

        self.update_table(self.SQL_Lite("get_data_class_feature"), [
                          "Class Name", "Feature"])

    def add_class_features(self):

        class_name = self.class_name.get()
        feature_name = self.feature_name.get()

        feature_type = ""
        if feature_name in [feature[0] for feature in
                            self.SQL_Lite("get_logic_features")]:
            feature_type = "logic"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_range_features")]:
            feature_type = "range"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_scalar_features")]:
            feature_type = "scalar"

        table_map = {
            "logic": "Class_Logic_Feature",
            "range": "Class_Range_Feature",
            "scalar": "Class_Scalar_Feature",
        }
        table_name = table_map.get(feature_type)

        if not table_name:

            return

        if (class_name, feature_name) not in self.SQL_Lite("get_data_class_feature"):
            self.SQL_Lite("send_to_"+str(table_name),
                          name=class_name, value1=feature_name)

        self.update_table(self.SQL_Lite("get_data_class_feature"), [
                          "Class Name", "Feature"])

    def show_class_feature_values(self):

        def update_feature_values(feature_name):

            feature_values = self.SQL_Lite(
                "get_feature_values", name=feature_name)
            flattened_values = []

            for value in feature_values:
                if len(value) == 1:
                    flattened_values.append(value[0])
                else:
                    flattened_values.extend(value)

            flattened_values = [str(i) for i in flattened_values]

            type = ""
            if feature_name in [row[0] for row in
                                self.SQL_Lite("get_range_values",
                                              name=feature_name)]:
                type = "range"
            elif feature_name in [row[0] for row in
                                  self.SQL_Lite("get_logic_values",
                                                name=feature_name)]:
                type = "logic"
            elif feature_name in [row[0] for row in
                                  self.SQL_Lite("get_scalar_values",
                                                name=feature_name)]:
                type = "scalar"

            if type == "range":

                if self.entry_feature_value:
                    self.entry_feature_value.destroy()
                    self.entry_feature_value = None
                self.entry_feature_value = customtkinter.CTkEntry(
                    self.main_frame)
                self.entry_feature_value.grid(
                    row=1, column=0, columnspan=2, padx=10, pady=5,
                    sticky="wne")
            else:

                if self.entry_feature_value:
                    self.entry_feature_value.destroy()
                    self.entry_feature_value = None
                self.entry_feature_value = customtkinter.CTkComboBox(
                    self.main_frame, values=flattened_values)
                self.entry_feature_value.grid(
                    row=1, column=0, columnspan=3, padx=10, pady=5,
                    sticky="wne")
                self.entry_feature_value.set("")
                self.entry_feature_value.configure(values=flattened_values)

        def update_class_features(class_name):

            feature_names = self.get_feature_names_where_class()
            self.feature_name.configure(values=feature_names)

            self.entry_feature_value.set("")

            self.feature_name.set("")

        self.frame_stack.append(self.current_buttons.copy())
        class_names = [row[0] for row in self.SQL_Lite("get_classes")]

        self.class_name = customtkinter.CTkComboBox(
            self.main_frame, values=class_names,
            command=lambda event: update_class_features(event))
        self.class_name.grid(row=0, column=0, padx=10, pady=5, sticky="wne")

        feature_names = self.get_feature_names_where_class()

        self.feature_name = customtkinter.CTkComboBox(
            self.main_frame, values=feature_names,
            command=lambda event: update_feature_values(event))
        self.feature_name.grid(row=0, column=1, padx=10, pady=5, sticky="wne")

        self.btn_add_class = customtkinter.CTkButton(
            self.main_frame, text="Добавить", command=self.add_class_feature)
        self.btn_add_class.grid(row=0, column=2, padx=10, pady=5, sticky="ne")

        self.entry_feature_value = None

        data = self.SQL_Lite("get_data")
        headers = ["Class Name", "Feature", "Value"]

        self.table_class_features = Table(
            self.main_frame, headers, data, height=300)
        self.table_class_features.grid(
            row=2, column=0, padx=10, pady=10, sticky="new", columnspan=3)
        self.update_table(data, headers)

    def get_feature_names_where_class(self):
        class_name = self.class_name.get()
        logic_features = [row[0] for row in self.SQL_Lite(
            "get_classes_logic_feature", name=class_name)]
        range_features = [row[0] for row in self.SQL_Lite(
            "get_classes_range_feature", name=class_name)]
        scalar_features = [row[0] for row in self.SQL_Lite(
            "get_classes_scalar_feature", name=class_name)]

        return logic_features + range_features + scalar_features

    def selected_feature(self, feature_selection):

        selected_feature = feature_selection
        feature_type = self.get_feature_type(selected_feature)

        if feature_type == "logic":
            data = self.SQL_Lite("get_logic_values", name=selected_feature)
            headers = ["Logic_Feature_Name", "Logic_Value0", "Logic_Value1"]
        elif feature_type == "range":
            data = self.SQL_Lite("get_range_values", name=selected_feature)
            headers = ["Range_Feature_Name",
                       "Range_Value_From", "Range_Value_To"]
        elif feature_type == "scalar":
            data = self.SQL_Lite("get_scalar_values", name=selected_feature)
            headers = ["Scalar_Feature_Name", "Scalar_Value"]
        else:
            return

        self.table_type_feature_values = Table(
            self.main_frame, headers, data, height=300)
        self.table_type_feature_values.grid(
            row=2, column=0, padx=10, pady=10, sticky="new", columnspan=3)

    def show_feature_values(self):
        self.frame_stack.append(self.current_buttons.copy())

        self.clear_main_frame()
        self.main_frame.grid_rowconfigure(3, weight=1)

        feature_names = {row[0] for row in self.get_feature_names()}
        feature_names = list(feature_names)
        self.all_feature_names = customtkinter.CTkComboBox(
            self.main_frame, values=feature_names,
            command=self.selected_feature)
        self.all_feature_names.grid(
            row=0, column=0, padx=10, pady=5, sticky="wne")

        self.write_feature_values = customtkinter.CTkButton(
            self.main_frame, text="Заполнить",
            command=self.handle_feature_input)
        self.write_feature_values.grid(
            row=0, column=1, padx=10, pady=5, sticky="wne")

        self.btn_delete_class = customtkinter.CTkButton(
            self.main_frame, text="Удалить", command=self.delete_feature_class)
        self.btn_delete_class.grid(
            row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.current_buttons = [self.btn_back]

    def delete_feature_class(self):

        feature = self.all_feature_names.get()

        self.SQL_Lite("clear_feature", name=feature)

    def handle_feature_input(self):

        selected_feature = self.all_feature_names.get()
        feature_type = self.get_feature_type(selected_feature)

        if feature_type == "logic":
            self.show_message(
                "Для логических значений нельзя вводить признаки")
        elif feature_type == "range":
            self.open_range_feature_form(name=selected_feature)
        elif feature_type == "scalar":
            self.open_scalar_feature_form()

    def show_range_input(self):
        self.range_min = customtkinter.CTkEntry(
            self.main_frame, placeholder_text="Значение от")
        self.range_min.grid(row=1, column=0, padx=10, pady=5, sticky="wne")

        self.range_max = customtkinter.CTkEntry(
            self.main_frame, placeholder_text="Значение до")
        self.range_max.grid(row=1, column=1, padx=10, pady=5, sticky="wne")

    def show_scalar_input(self):
        self.scalar_entries = []
        self.add_scalar_entry()
        self.add_value_button = customtkinter.CTkButton(
            self.main_frame, text="Добавить", command=self.add_scalar_entry)
        self.add_value_button.grid(
            row=2, column=0, padx=10, pady=5, sticky="wne")

    def add_scalar_entry(self):
        entry = customtkinter.CTkEntry(self.main_frame)
        entry.grid(row=3 + len(self.scalar_entries),
                   column=0, padx=10, pady=5, sticky="wne")
        self.scalar_entries.append(entry)

    def show_message(self, message):
        msg_window = customtkinter.CTkToplevel(self)
        msg_window.title("Сообщение")
        label = customtkinter.CTkLabel(msg_window, text=message)
        label.pack(padx=20, pady=20)
        button = customtkinter.CTkButton(
            msg_window, text="OK", command=msg_window.destroy)
        button.pack(pady=10)

    def get_feature_type(self, feature_name):
        feature_type = ""
        if feature_name in [feature[0] for feature in
                            self.SQL_Lite("get_logic_features")]:
            feature_type = "logic"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_range_features")]:
            feature_type = "range"
        elif feature_name in [feature[0] for feature in
                              self.SQL_Lite("get_scalar_features")]:
            feature_type = "scalar"

        return feature_type

    def check_fillness(self):

        classes = {i[0] for i in self.SQL_Lite("get_classes")}

        missing_info = []

        range_features = self.SQL_Lite("get_range_feature_value_rows")
        scalar_features = self.SQL_Lite("get_scalar_feature_value_rows")

        class_logic_features = self.SQL_Lite("get_class_logic_features")
        class_range_features = self.SQL_Lite("get_class_range_features")
        class_scalar_features = self.SQL_Lite("get_class_scalar_features")

        for row in range_features:
            if row[1] == None or row[1] == '':
                missing_info.append(
                    f"Признак '{row[0]}' - размерное значение From: {row[1]}")
            if row[2] == None or row[2] == '':
                missing_info.append(
                    f"Признак '{row[0]}' - размерное значение To: {row[2]}")

        for row in scalar_features:
            if row[1] == None or row[1] == '':
                missing_info.append(
                    f"Признак '{row[0]}' - скалярное значение: {row[1]}")

        classes_with_logic = [item[0] for item in class_logic_features]
        classes_with_range = [item[0] for item in class_range_features]
        classes_with_scalar = [item[0] for item in class_scalar_features]

        all_classes_with_features = set(classes_with_logic) | set(
            classes_with_range) | set(classes_with_scalar)
        classes_without_features = list(
            set(classes) - all_classes_with_features)

        for f in classes_without_features:
            missing_info.append(
                f"Класс '{f}' - отсутствует хотя бы один признак")

        missing = ""
        for row in class_range_features:
            if row[2] is None or row[2] == "":
                missing += f"Класс {row[0]} пропущенное значение признака {row[1]}\n"
                missing_info.append(missing)

        missing = ""
        for row in class_scalar_features:
            if row[2] is None or row[2] == "":
                missing += f"Класс {row[0]} пропущенное значение признака {row[1]}\n"
                missing_info.append(missing)

        result_text = "Проверка пройдена" if not missing_info else "\n".join(
            missing_info)
        if result_text == "Проверка пройдена":
            self.fillness_flag = True
        else:
            self.fillness_flag = False

        self.show_message(result_text)

    def show_message(self, text):
        window = customtkinter.CTkToplevel(self)
        window.title("Результат проверки")
        label = customtkinter.CTkLabel(window, text=text, wraplength=400)
        label.pack(padx=20, pady=20)
        button = customtkinter.CTkButton(
            window, text="ОК", command=window.destroy)
        button.pack(pady=10)

    def show_empty_entry(self, text):
        window = customtkinter.CTkToplevel(self)
        window.title("Ошибка")
        label = customtkinter.CTkLabel(window, text=text, wraplength=400)
        label.pack(padx=20, pady=20)
        button = customtkinter.CTkButton(
            window, text="ОК", command=window.destroy)
        button.pack(pady=10)

    def show_class_solve(self, text):
        window = customtkinter.CTkToplevel(self)
        window.title("Класс")
        label = customtkinter.CTkLabel(window, text=text, wraplength=400)
        label.pack(padx=20, pady=20)
        button = customtkinter.CTkButton(
            window, text="ОК", command=window.destroy)
        button.pack(pady=10)

    def show_previous_buttons(self):

        if self.frame_stack:
            previous_buttons = self.frame_stack.pop()
            self.Hide_current_elements()
            self.clear_main_frame()
            for widget in previous_buttons:
                widget.grid(row=previous_buttons.index(widget),
                            column=0, padx=10, pady=10, sticky="nsew")
            self.current_buttons = previous_buttons.copy()

    def Hide_current_elements(self):
        for widget in self.current_buttons:
            widget.grid_forget()

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def clear_active_frame(self):
        for widget in self.active_frame.winfo_children():
            widget.destroy()

    def add_class(self, name=None):
        self.class_listbox.configure(state="normal")
        new_class = self.entry_class.get().strip()
        if new_class and new_class not in self.class_list:
            self.SQL_Lite("send_class", name=new_class)
            self.class_list.append(new_class)
            self.update_class_listbox()
        self.class_listbox.configure(state="disabled")
        self.entry_class.delete(0, 'end')

    def delete_class(self):
        self.class_listbox.configure(state="normal")
        try:
            selected_text = self.class_listbox.get(
                "sel.first", "sel.last").strip()
        except TclError:
            selected_text = None
            self.class_listbox.configure(state="disabled")

        if selected_text in self.class_list:
            self.class_list.remove(selected_text)

            self.SQL_Lite("Delete_Class", value1=selected_text)
            self.update_class_listbox()
        self.class_listbox.configure(state="disabled")

    def update_class_listbox(self):
        self.class_listbox.delete("1.0", "end")
        for item in self.class_list:
            self.class_listbox.insert("end", item + "\n")

    def add_logic_feature(self):
        self.logic_feature_listbox.configure(state="normal")
        name = self.entry_logic_feature.get().strip()
        self.entry_logic_feature.delete(0, len(self.entry_logic_feature.get()))

        if (name not in self.logic_features) and (name != None) and (name != ""):
            self.SQL_Lite("send_logic_feature", name=name)
            self.logic_features.append(name)
            self.update_logic_feature_listbox()
        self.logic_feature_listbox.configure(state="disabled")

        self.update_logic_feature_listbox()

    def delete_logic_feature(self):
        self.logic_feature_listbox.configure(state="normal")
        try:
            selected_text = self.logic_feature_listbox.get(
                "sel.first", "sel.last").strip()

            if selected_text in self.logic_features:
                self.logic_features.remove(selected_text)
                self.SQL_Lite("Delete_logic_feature", value1=selected_text)
                self.update_logic_feature_listbox()
            self.logic_feature_listbox.configure(state="disabled")
        except TclError:
            selected_text = None
            self.logic_feature_listbox.configure(state="disabled")

    def update_logic_feature_listbox(self):
        self.logic_feature_listbox.delete("1.0", "end")
        for item in self.logic_features:
            self.logic_feature_listbox.insert("end", item + "\n")

    def add_range_feature(self):
        self.range_feature_listbox.configure(state="normal")
        name = self.entry_range_feature.get().strip()
        self.entry_range_feature.delete(0, len(self.entry_range_feature.get()))

        if (name not in self.range_features) and (name != None) and (name != ""):
            self.SQL_Lite("send_range_feature", name=name)
            self.range_features.append(name)
            self.update_range_feature_listbox()
        self.range_feature_listbox.configure(state="disabled")
        self.update_range_feature_listbox()

    def update_range_feature(self):
        name = self.label_feature_name.cget("text").strip()
        value1 = self.entry_feature_value1.get().strip()
        value2 = self.entry_feature_value2.get().strip()

        if value1 == '' or value2 == '':
            self.show_empty_entry(text="заполните значения")
        elif int(value1) > int(value2):

            self.form_window.destroy()
            self.show_message(text="Неверный диапазон")

        else:
            self.SQL_Lite("update_range_feature", name=name,
                          value1=value1, value2=value2)
            self.form_window.destroy()

    def delete_range_feature(self):
        self.range_feature_listbox.configure(state="normal")
        try:
            selected_text = self.range_feature_listbox.get(
                "sel.first", "sel.last").strip()

            if selected_text in self.range_features:
                self.range_features.remove(selected_text)
                self.SQL_Lite("Delete_range_feature", value1=selected_text)
                self.update_range_feature_listbox()
            self.range_feature_listbox.configure(state="disabled")
        except TclError:
            selected_text = None
            self.range_feature_listbox.configure(state="disabled")

    def update_range_feature_listbox(self):
        self.range_feature_listbox.delete("1.0", "end")

        for item in self.range_features:
            self.range_feature_listbox.insert("end", item + "\n")

    def add_scalar_feature(self):
        self.scalar_feature_listbox.configure(state="normal")
        name = self.entry_scalar_feature.get().strip()

        if (name not in self.scalar_features) and (name != None) and (name != ""):
            self.SQL_Lite("send_scalar_feature", name=name)
            self.scalar_features.append(name)
            self.update_scalar_feature_listbox()
        self.scalar_feature_listbox.configure(state="disabled")
        self.update_scalar_feature_listbox()

    def delete_scalar_feature(self):
        self.scalar_feature_listbox.configure(state="normal")

        try:
            selected_text = self.scalar_feature_listbox.get(
                "sel.first", "sel.last").strip()

            if selected_text in self.scalar_features:
                self.scalar_features.remove(selected_text)
                self.SQL_Lite("Delete_scalar_feature", value1=selected_text)
                self.update_scalar_feature_listbox()
            self.scalar_feature_listbox.configure(state="disabled")
        except TclError:
            selected_text = None
            self.scalar_feature_listbox.configure(state="disabled")

    def update_scalar_feature_listbox(self):
        self.scalar_feature_listbox.delete("1.0", "end")
        for item in self.scalar_features:
            self.scalar_feature_listbox.insert("end", item + "\n")

    def update_table(self, data, header):
        self.data = self.SQL_Lite("get_data")
        self.table_class_features.destroy()
        self.table_class_features = Table(
            self.main_frame, header, data, height=300)
        self.table_class_features.grid(
            row=2, column=0, padx=10, pady=10, sticky="new", columnspan=3)

        self.data = self.SQL_Lite("get_data")

    def add_class_feature(self):

        if self.entry_feature_value.get() != "":

            class_name = self.class_name.get()
            feature_name = self.feature_name.get()
            value = self.entry_feature_value.get()

            feature_type = ""
            if feature_name in [feature[0] for feature in
                                self.SQL_Lite("get_logic_features")]:
                feature_type = "logic"
            elif feature_name in [feature[0] for feature in
                                  self.SQL_Lite("get_range_features")]:
                feature_type = "range"
            elif feature_name in [feature[0] for feature in
                                  self.SQL_Lite("get_scalar_features")]:
                feature_type = "scalar"

            table_map = {
                "logic": "Class_Logic_Feature",
                "range": "Class_Range_Feature",
                "scalar": "Class_Scalar_Feature",
            }
            table_name = table_map.get(feature_type)

            if not table_name:
                return

            if self.entry_feature_value is customtkinter.CTkEntry:
                if self.SQL_Lite("get_range_bool", name=feature_name,
                                 value1=value):
                    self.SQL_Lite(
                        "update_"+str(table_name), name=class_name,
                        value1=feature_name, value2=value)
                    self.update_table(self.SQL_Lite("get_data"), [
                                      "Class", "Feature", "Value"])
            else:

                self.SQL_Lite("update_"+str(table_name), name=class_name,
                              value1=feature_name, value2=value)

                self.update_table(self.SQL_Lite("get_data"), [
                                  "Class", "Feature", "Value"])

        else:
            self.show_empty_entry(text="Введите значение признака")

    def _class_feature(self):

        if self.entry_feature_value.get() != "":

            class_name = self.class_name.get()
            feature_name = self.feature_name.get()

            feature_type = ""
            if feature_name in [feature[0] for feature in
                                self.SQL_Lite("get_logic_features")]:
                feature_type = "logic"
            elif feature_name in [feature[0] for feature in
                                  self.SQL_Lite("get_range_features")]:
                feature_type = "range"
            elif feature_name in [feature[0] for feature in
                                  self.SQL_Lite("get_scalar_features")]:
                feature_type = "scalar"

            table_map = {
                "logic": "Class_Logic_Feature",
                "range": "Class_Range_Feature",
                "scalar": "Class_Scalar_Feature",
            }
            table_name = table_map.get(feature_type)

            if not table_name:

                return

            self.SQL_Lite("send_to_"+str(table_name),
                          name=class_name, value1=feature_name)

            self.update_table(self.SQL_Lite("get_data"))

        else:
            self.show_empty_entry(text="Введите значение признака")

    def Solve_Task(self):
        self.check_fillness()

        if self.fillness_flag:

            self.frame_stack.append(self.current_buttons.copy())

            self.main_frame = customtkinter.CTkScrollableFrame(
                self.active_frame)
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(8, weight=1)
            self.main_frame.grid(row=1, column=0, padx=5,
                                 pady=5, sticky="nsew")

            self.back_frame = customtkinter.CTkFrame(self.active_frame)
            self.back_frame.grid_columnconfigure(4, weight=1)
            self.back_frame.grid_rowconfigure(0, weight=1)
            self.back_frame.grid(row=0, column=0, padx=5,
                                 pady=5, sticky="nsew")

            self.btn_back = customtkinter.CTkButton(
                self.back_frame, text="Назад",
                command=self.show_previous_buttons)
            self.btn_back.grid(row=2, column=0, padx=10, pady=10, sticky="w")

            self.selected_logic_features = []
            self.selected_logic_entries = []

            self.selected_scalar_features = []
            self.selected_scalar_entries = []

            self.selected_range_features = []
            self.selected_range_entries = []

            self.logic_feature_label = customtkinter.CTkLabel(
                self.main_frame, text="Логические признаки:")
            self.logic_feature_label.grid(
                row=2, column=0, padx=10, pady=5, sticky="w")

            self.logic_feature_listbox = customtkinter.CTkComboBox(
                self.main_frame, values=[i[0] for i in
                                         self.SQL_Lite("get_logic_features")],
                command=self.add_logic_feature_solve)
            self.logic_feature_listbox.grid(
                row=3, column=0, padx=10, pady=5, sticky="nwe")

            self.logic_frame = customtkinter.CTkFrame(self.main_frame)
            self.logic_frame.grid(
                row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nwe")

            self.scalar_feature_label = customtkinter.CTkLabel(
                self.main_frame, text="Скалярные признаки:")
            self.scalar_feature_label.grid(
                row=5, column=0, padx=10, pady=5, sticky="w")

            self.scalar_feature_listbox = customtkinter.CTkComboBox(
                self.main_frame, values=list({i[0] for i in
                                         self.SQL_Lite("get_scalar_features")}),
                command=self.add_scalar_feature_solve)
            self.scalar_feature_listbox.grid(
                row=6, column=0, padx=10, pady=5, sticky="nwe")

            self.scalar_frame = customtkinter.CTkFrame(self.main_frame)
            self.scalar_frame.grid(
                row=7, column=0, columnspan=2, padx=10, pady=5, sticky="nwe")

            self.range_feature_label = customtkinter.CTkLabel(
                self.main_frame, text="Размерные признаки:")
            self.range_feature_label.grid(
                row=8, column=0, padx=10, pady=5, sticky="w")

            self.range_feature_listbox = customtkinter.CTkComboBox(
                self.main_frame, values=[i[0] for i in
                                         self.SQL_Lite("get_range_features")],
                command=self.add_range_feature_solve)
            self.range_feature_listbox.grid(
                row=9, column=0, padx=10, pady=5, sticky="nwe")

            self.range_frame = customtkinter.CTkFrame(self.main_frame)
            self.range_frame.grid(
                row=10, column=0, columnspan=2, padx=10, pady=5, sticky="nwe")

            self.btn_submit = customtkinter.CTkButton(
                self.active_frame, text="Решить", command=self.get_task_value)
            self.btn_submit.grid(row=8, column=0, padx=10,
                                 pady=5, sticky="nwe")

            self.current_buttons = self.default_buttons.copy()

        else:
            self.show_message(text="полнота не пройдена")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path.set(file_path)

    def predict_class(self):
        if not self.image_path.get() or self.image_path.get() == "Изображение не выбрано":
            return
        predicted_class = Main_Predict(self.image_path.get())

        class_map = {
            0: "Academic",
            1: "Caligraphic",
            2: "Expression",
            3: "Haus",
            4: "Individual",
            5: "Machinelike",
        }
        predicted_class = class_map.get(predicted_class)

        self.show_predicted_class(text=predicted_class)

    def add_to_training(self):

        src = self.image_path.get()

        text = add_to_model(src)
        self.show_message(text=text)

    def show_predicted_class(self, text):

        window = customtkinter.CTkToplevel(self)
        window.title("Результат проверки")
        label = customtkinter.CTkLabel(window, text="Предсказанный класс: " +
                                       text + "\nНажмите добавить если правильный класс", wraplength=400)
        label.pack(padx=20, pady=20)

        button = customtkinter.CTkButton(
            window, text="ОК", command=window.destroy)
        button.pack(pady=10)

        button_add = customtkinter.CTkButton(
            window, text="Добавить", command=self.add_to_training)
        button_add.pack(pady=10)

    def get_task_value(self):
        all_classes = set(row[0] for row in self.SQL_Lite("get_classes"))

        logic_values = {
            feature: entry.get()
            for feature, entry in self.selected_logic_entries
            if entry.winfo_exists()
        }

        scalar_values = {
            feature: entry.get()
            for feature, entry in self.selected_scalar_entries
            if entry.winfo_exists()
        }

        range_values = {
            feature: var.get()
            for feature, var in self.selected_range_features
        }

        excluded_classes = {}

        # Опровержения по логическим признакам
        for feature, value in logic_values.items():
            matching = set(row[0] for row in self.SQL_Lite(
                "get_matched_classes_logic", name=feature, value1=value))

            present = set(row[0] for row in self.SQL_Lite(
                "get_classes_with_logic_feature", name=feature))

            for cls in all_classes:
                if cls not in present:
                    excluded_classes.setdefault(cls, []).append(
                        f"отсутствует логический признак '{feature}'")
                elif cls not in matching:
                    excluded_classes.setdefault(cls, []).append(
                        f"логический признак '{feature}' != '{value}'")

        # # Опровержения по скалярам
        for feature, value in scalar_values.items():
            matching = set(row[0] for row in self.SQL_Lite(
                "get_matched_classes_scalar", name=feature, value1=value))

            present = set(row[0] for row in self.SQL_Lite(
                "get_classes_with_scalar_feature", name=feature))

            for cls in all_classes:
                if cls not in present:
                    excluded_classes.setdefault(cls, []).append(
                        f"отсутствует скалярный признак '{feature}'")
                elif cls not in matching:
                    excluded_classes.setdefault(cls, []).append(
                        f"скалярный признак '{feature}' != '{value}'")

        # Опровержения по диапазонам
        for feature, value in range_values.items():
            matching = set(row[0] for row in self.SQL_Lite(
                "get_matched_classes_range", name=feature, value1=value))

            present = set(row[0] for row in self.SQL_Lite(
                "get_classes_with_range_feature", name=feature))

            for cls in all_classes:
                if cls not in present:
                    excluded_classes.setdefault(cls, []).append(
                        f"отсутствует размерный признак '{feature}'")
                elif cls not in matching:
                    excluded_classes.setdefault(cls, []).append(
                        f"размерный признак '{feature}' не соответствует значению '{value}'")

        possible_classes = all_classes - excluded_classes.keys()

        if possible_classes:
            class_name = ", ".join(possible_classes)

            reason_text = f"Определённый класс почерка: {class_name}\n\n"
            other_exclusions = {
                cls: reasons for cls, reasons in excluded_classes.items()
                if cls not in possible_classes
            }

            if other_exclusions:
                reason_text += "Остальные классы были исключены по следующим причинам:\n"
                for cls, reasons in other_exclusions.items():
                    reason_text += f"- Класс '{cls}' исключён, так как: {', '.join(reasons)}\n"

            self.show_class_solve(text=reason_text.strip())

        else:
            # Никто не подошёл
            reason_text = "Не удалось определить класс почерка.\nПричины:\n"
            for cls, reasons in excluded_classes.items():
                reason_text += f"- Класс '{cls}' исключён, так как: {', '.join(reasons)}\n"
            self.show_class_solve(text=reason_text.strip())

        return class_name if possible_classes else None

    def add_logic_feature_solve(self, feature):
        if feature not in [f[0] for f in self.selected_logic_features]:
            self.selected_logic_features.append((feature, ["Да", "Нет"]))
            self.update_feature_entries_solve(
                self.logic_frame, self.selected_logic_features, values=["Да", "Нет"])

    def add_scalar_feature_solve(self, feature):
        if feature not in [f[0] for f in self.selected_scalar_features]:
            var = customtkinter.StringVar()
            values = [row[1] for row in self.SQL_Lite(
                "get_scalar_values", name=feature)]
            self.selected_scalar_features.append((feature, var))
            self.update_feature_entries_solve(
                self.scalar_frame, self.selected_scalar_features, values=values)

    def add_range_feature_solve(self, feature):
        if feature not in [f[0] for f in self.selected_range_features]:
            var = customtkinter.StringVar()
            self.selected_range_features.append((feature, var))
            self.update_feature_entries_solve(
                self.range_frame, self.selected_range_features)

    def update_feature_entries_solve(self, frame, features, values=None):
        for widget in frame.winfo_children():
            widget.destroy()

        for i, (feature, var) in enumerate(features):
            label = customtkinter.CTkLabel(frame, text=feature)
            label.grid(row=i, column=0, padx=5, pady=2, sticky="w")

            if frame == self.logic_frame:
                entry = customtkinter.CTkComboBox(frame, values=values) if values else customtkinter.CTkEntry(
                    frame, textvariable=var, placeholder_text="Введите значение")
                entry.grid(row=i, column=1, padx=5, pady=2, sticky="we")
                self.selected_logic_entries.append((feature, entry))
                btn = customtkinter.CTkButton(frame, text="Удалить", width=60,
                                            command=lambda f=feature: self.remove_feature_solve(f, "logic"))
                btn.grid(row=i, column=2, padx=5, pady=2)

            elif frame == self.range_frame:
                entry = customtkinter.CTkComboBox(frame, values=values) if values else customtkinter.CTkEntry(
                    frame, textvariable=var, placeholder_text="Введите значение")
                entry.grid(row=i, column=1, padx=5, pady=2, sticky="we")
                self.selected_range_entries.append((feature, entry))
                btn = customtkinter.CTkButton(frame, text="Удалить", width=60,
                                            command=lambda f=feature: self.remove_feature_solve(f, "range"))
                btn.grid(row=i, column=2, padx=5, pady=2)

            elif frame == self.scalar_frame:
                values = [row[1] for row in self.SQL_Lite("get_scalar_values", name=feature)]
                entry = customtkinter.CTkComboBox(frame, values=values) if values else customtkinter.CTkEntry(
                    frame, textvariable=var, placeholder_text="Введите значение")
                entry.grid(row=i, column=1, padx=5, pady=2, sticky="we")
                self.selected_scalar_entries.append((feature, entry))
                btn = customtkinter.CTkButton(frame, text="Удалить", width=60,
                                            command=lambda f=feature: self.remove_feature_solve(f, "scalar"))
                btn.grid(row=i, column=2, padx=5, pady=2)

    def remove_feature_solve(self, feature, feature_type):
        if feature_type == "logic":
            self.selected_logic_features = [f for f in self.selected_logic_features if f[0] != feature]
            self.selected_logic_entries = [e for e in self.selected_logic_entries if e[0] != feature]
            self.update_feature_entries_solve(self.logic_frame, self.selected_logic_features, values=["Да", "Нет"])

        elif feature_type == "scalar":
            self.selected_scalar_features = [f for f in self.selected_scalar_features if f[0] != feature]
            self.selected_scalar_entries = [e for e in self.selected_scalar_entries if e[0] != feature]
            self.update_feature_entries_solve(self.scalar_frame, self.selected_scalar_features)

        elif feature_type == "range":
            self.selected_range_features = [f for f in self.selected_range_features if f[0] != feature]
            self.selected_range_entries = [e for e in self.selected_range_entries if e[0] != feature]
            self.update_feature_entries_solve(self.range_frame, self.selected_range_features)


    def write_feature_values(self):
        ...

    def open_range_feature_form(self, name):
        self.form_window = customtkinter.CTkToplevel(self)
        self.form_window.wm_attributes("-topmost", True)
        self.form_window.title("Добавить размерный признак")
        self.form_window.geometry("350x150")
        self.form_window.columnconfigure(1, weight=1)

        self.label_feature_name = customtkinter.CTkLabel(
            self.form_window, text=name)
        self.label_feature_name.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="we")

        self.entry_feature_value1 = customtkinter.CTkEntry(
            self.form_window, placeholder_text="от")
        self.entry_feature_value1.grid(
            row=1, column=0, padx=10, pady=5, sticky="w")

        self.entry_feature_value2 = customtkinter.CTkEntry(
            self.form_window, placeholder_text="до")
        self.entry_feature_value2.grid(
            row=1, column=1, padx=10, pady=5, sticky="e")

        self.btn_add_feature = customtkinter.CTkButton(
            self.form_window, text="Добавить",
            command=self.update_range_feature)
        self.btn_add_feature.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="e")

    def update_scalar_feature(self, name, values):
        for value in values:
            self.SQL_Lite("send_scalar_value", name=name, value1=value)

        self.form_window.destroy()

    def open_scalar_feature_form(self):

        def collect_values():
            values = [entry.get().strip()
                      for entry in self.value_entries if entry.get().strip()]
            if values:

                self.update_scalar_feature(
                    self.all_feature_names.get(), values)

        self.form_window = customtkinter.CTkToplevel(self)
        self.form_window.wm_attributes("-topmost", True)
        self.form_window.title("Добавить скалярный признак")
        self.form_window.geometry("350x150")
        self.form_window.columnconfigure(1, weight=1)

        self.entry_feature_name = customtkinter.CTkLabel(
            self.form_window, text=self.all_feature_names.get())
        self.entry_feature_name.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="we")

        self.value_entries = []

        self.entry_feature_value1 = customtkinter.CTkEntry(self.form_window)
        self.entry_feature_value1.grid(
            row=1, column=0, columnspan=2, padx=10, pady=5, sticky="we")

        self.btn_add_feature = customtkinter.CTkButton(
            self.form_window, text="Добавить", command=collect_values)
        self.btn_add_feature.grid(
            row=100, column=0, columnspan=2, padx=10, pady=5, sticky="e")

        def add_value_entry():
            entry = customtkinter.CTkEntry(self.form_window)
            entry.grid(row=len(self.value_entries) + 1, column=0,
                       columnspan=2, padx=10, pady=5, sticky="we")
            self.value_entries.append(entry)

        add_value_entry()

        self.btn_add_field = customtkinter.CTkButton(
            self.form_window, text="Добавить поле", command=add_value_entry)
        self.btn_add_field.grid(
            row=99, column=0, columnspan=2, padx=10, pady=5, sticky="we")

    def get_scalar_names(self):
        return self.SQL_Lite("get_scalar_features")

    def get_range_names(self):
        return self.SQL_Lite("get_range_features")

    def get_logic_names(self):
        return self.SQL_Lite("get_logic_features")

    def get_feature_names(self):
        return self.SQL_Lite("get_features")

    def get_class_names(self):

        return self.SQL_Lite("get_classes")

    def SQL_Lite(self, task, name=None, value1=None, value2=None):
        connection = sqlite3.connect('db.sqlite')
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        rows = []
        if "get" in task:
            if task == "get_class_logic_features":
                cursor.execute("SELECT * FROM Class_Logic_Features")
                rows = cursor.fetchall()
                return rows
            if task == "get_class_range_features":
                cursor.execute("SELECT * FROM Class_Range_Feature")
                rows = cursor.fetchall()
                return rows
            if task == "get_class_scalar_features":
                cursor.execute("SELECT * FROM Class_Scalar_Feature")
                rows = cursor.fetchall()
                return rows

            if task == "get_range_feature_value_rows":
                cursor.execute(
                    "SELECT Range_Feature_Name, Range_Value_From, Range_Value_To FROM Range_Features")
                rows = cursor.fetchall()
                return rows

            if task == "get_scalar_feature_value_rows":
                cursor.execute(
                    "SELECT Scalar_Feature_Name, Scalar_Feature_Values  FROM Scalar_Features")
                rows = cursor.fetchall()
                return rows

            if task == "get_range_values":
                cursor.execute(
                    "SELECT Range_Feature_Name, Range_Value_From, Range_Value_To FROM Range_Features WHERE Range_Feature_Name = ?", (name,))
                rows = cursor.fetchall()
                return rows
            if task == "get_logic_values":
                cursor.execute(
                    "SELECT Logic_Feature_Name, Logic_Value0, Logic_value1 FROM Logic_Features WHERE Logic_Feature_Name = ?", (name,))
                rows = cursor.fetchall()
                return rows
            if task == "get_scalar_values":
                cursor.execute(
                    "SELECT Scalar_Feature_Name, Scalar_Feature_Values FROM Scalar_Features WHERE Scalar_Feature_Name = ?", (name,))
                rows = cursor.fetchall()
                return rows

            if task == "get_range_bool":

                cursor.execute(
                    "SELECT Range_Feature_Name, Range_Value_From, Range_Value_To FROM Range_Features WHERE Range_Feature_Name = ?", (name,))
                rows = cursor.fetchall()

                if rows[0][1] <= int(value1) and int(value1) <= rows[0][2]:

                    return True
                else:

                    return False

            if task == "get_feature_values":

                data = []

                tables = [
                    ("Logic_Features", "Logic_Feature_Name",
                     ["Logic_Value0", "Logic_value1"]),
                    ("Range_Features", "Range_Feature_Name",
                     ["Range_Value_From", "Range_Value_To"]),
                    ("Scalar_Features", "Scalar_Feature_Name",
                     ["Scalar_Feature_Values"]),
                ]

                for table, feature, feature_column in tables:
                    query = f"SELECT {', '.join(feature_column)} FROM {table} WHERE {feature} = ?"

                    cursor.execute(query, (name,))
                    data.extend(cursor.fetchall())

                return data

            if task == "get_empty_logic_features":
                cursor.execute(
                    "SELECT Logic_Feature_Name FROM Logic_Features WHERE Logic_Value0 IS NULL OR Logic_Value0 = '' OR Logic_Value1 IS NULL OR Logic_Value1 = '' ")
                rows = cursor.fetchall()
                return rows
            if task == "get_empty_range_features":
                cursor.execute(
                    "SELECT Range_Feature_Name FROM Range_Features WHERE Range_Value_From IS NULL OR Range_Value_To = '' OR Range_Value_From IS NULL OR Range_Value_To = '' ")
                rows = cursor.fetchall()
                return rows

            if task == "get_empty_scalar_features":
                cursor.execute(
                    "SELECT Scalar_Feature_Name FROM Scalar_Features WHERE Scalar_Feature_Values IS NULL OR Scalar_Feature_Values = '' ")
                rows = cursor.fetchall()
                return rows

            if task == "get_logic_features":
                cursor.execute("SELECT Logic_Feature_Name FROM Logic_Features")
                rows = cursor.fetchall()
                return rows
            if task == "get_range_features":
                cursor.execute("SELECT Range_Feature_Name FROM Range_Features")
                rows = cursor.fetchall()
                return rows
            if task == "get_scalar_features":
                cursor.execute(
                    "SELECT Scalar_Feature_Name FROM Scalar_Features")
                rows = cursor.fetchall()
                return rows
            if task == "get_features":
                cursor.execute("SELECT Logic_Feature_Name FROM Logic_Features")
                rows = cursor.fetchall()
                cursor.execute("SELECT Range_Feature_Name FROM Range_Features")
                rows.extend(cursor.fetchall())
                cursor.execute(
                    "SELECT Scalar_Feature_Name FROM Scalar_Features")
                rows.extend(cursor.fetchall())
                return rows
            if task == "get_classes":
                cursor.execute("SELECT Class_Name FROM Classes")
                rows = cursor.fetchall()

                return rows
            if task == "get_data":

                data = []

                tables = [
                    ("Class_Logic_Features", "Logic_Feature"),
                    ("Class_Range_Feature", "Range_Feature"),
                    ("Class_Scalar_Feature", "Scalar_Feature"),
                ]

                for table, feature_column in tables:
                    cursor.execute(
                        f"SELECT Class_Name, {feature_column}, Value FROM {table}")
                    data.extend(cursor.fetchall())

                return data
            if task == "get_data_class_feature":

                data = []

                tables = [
                    ("Class_Logic_Features", "Logic_Feature"),
                    ("Class_Range_Feature", "Range_Feature"),
                    ("Class_Scalar_Feature", "Scalar_Feature"),
                ]

                for table, feature_column in tables:
                    cursor.execute(
                        f"SELECT Class_Name, {feature_column} FROM {table}")
                    data.extend(cursor.fetchall())

                return data

            if task == "get_classes_logic_feature":
                cursor.execute(
                    "SELECT Logic_Feature FROM Class_Logic_Features WHERE Class_Name = ?", (name,))
                rows = cursor.fetchall()

                return rows
            if task == "get_classes_range_feature":
                cursor.execute(
                    "SELECT Range_Feature FROM Class_Range_Feature WHERE Class_Name = ?", (name,))
                rows = cursor.fetchall()
                return rows
            if task == "get_classes_scalar_feature":
                cursor.execute(
                    "SELECT Scalar_Feature FROM Class_Scalar_Feature WHERE Class_Name = ?", (name,))
                rows = cursor.fetchall()
                return rows
            if task == "get_matched_classes_logic":
                cursor.execute(
                    "SELECT Class_Name FROM Class_Logic_Features WHERE Logic_Feature = ? AND Value = ?", (name, value1))
                rows = cursor.fetchall()
                return rows
            if task == "get_matched_classes_range":
                cursor.execute(
                    "SELECT Class_Name FROM Class_Range_Feature WHERE Range_Feature = ? AND Value = ?", (name, value1))
                rows = cursor.fetchall()
                return rows
            if task == "get_matched_classes_scalar":
                cursor.execute(
                    "SELECT Class_Name FROM Class_Scalar_Feature WHERE Scalar_Feature = ? AND Value = ?", (name, value1))
                rows = cursor.fetchall()
                return rows
            if task == "get_classes_with_logic_feature":
                cursor.execute(
                    "SELECT DISTINCT Class_Name FROM Class_Logic_Features WHERE Logic_Feature = ?", (name,))
                rows = cursor.fetchall()
                return rows
            if task == "get_classes_with_scalar_feature":
                cursor.execute(
                    "SELECT DISTINCT Class_Name FROM Class_Scalar_Feature WHERE Scalar_Feature = ?", (name,))
                rows = cursor.fetchall()
                return rows

            if task == "get_classes_with_range_feature":
                cursor.execute(
                    "SELECT DISTINCT Class_Name FROM Class_Range_Feature WHERE Range_Feature = ?", (name,))
                rows = cursor.fetchall()
                return rows
        if "remove" in task:

            if "Class_Logic_Feature" in task:
                cursor.execute(
                    "DELETE FROM Class_Logic_Feature WHERE Scalar_Feature_Name is ? AND Scalar_Feature_Values IS NULL", (name,))
                connection.commit()
                return rows

            if "Class_Range_Feature" in task:
                ...

            if "Class_Scalar_Feature" in task:
                ...

        elif "send" in task:
            if task == "send_scalar_value":
                cursor.execute(
                    "INSERT INTO Scalar_Features (Scalar_Feature_Name, Scalar_Feature_Values) VALUES (?, ?);", (name, value1))
                cursor.execute(
                    "DELETE FROM Scalar_Features WHERE Scalar_Feature_Name is ? AND Scalar_Feature_Values IS NULL", (name,))
                connection.commit()
                return rows
            if task == "send_logic_feature":
                cursor.execute(
                    "INSERT INTO Logic_Features (Logic_Feature_Name) VALUES (?)", (name,))
                connection.commit()
                return rows
            elif task == "send_range_feature":
                cursor.execute(
                    "INSERT INTO Range_Features (Range_Feature_Name) VALUES (?)", (name, ))
                connection.commit()
                return rows
            elif task == "send_scalar_feature":
                cursor.execute(
                    "INSERT INTO Scalar_Features (Scalar_Feature_Name) VALUES (?)", (name,))
                connection.commit()
                return rows
            elif task == "send_class":
                cursor.execute(
                    "INSERT INTO Classes (Class_Name) VALUES (?)", (name,))
                connection.commit()
                return rows
            elif task == "send_to_Class_Logic_Feature":

                cursor.execute(
                    "INSERT INTO Class_Logic_Features (Class_Name, Logic_Feature) VALUES (?,?)", (name, value1,))
                connection.commit()
                return rows
            elif task == "send_to_Class_Range_Feature":

                cursor.execute(
                    "INSERT INTO Class_Range_Feature (Class_Name, Range_Feature) VALUES (?,?)", (name, value1, ))
                connection.commit()
                return rows
            elif task == "send_to_Class_Scalar_Feature":

                cursor.execute(
                    "INSERT INTO Class_Scalar_Feature (Class_Name, Scalar_feature) VALUES (?,?)", (name, value1))
                connection.commit()
                return rows
        elif "Delete" in task:
            if task == "Delete_Class":
                cursor.execute(
                    "DELETE FROM Class_Logic_Features WHERE Class_Name = ?", (value1,))
                cursor.execute(
                    "DELETE FROM Class_Scalar_Feature WHERE Class_Name = ?", (value1,))
                cursor.execute(
                    "DELETE FROM Class_Range_Feature WHERE Class_Name = ?", (value1,))
                cursor.execute(
                    "DELETE FROM Classes WHERE Class_Name =?", (value1,))
                connection.commit()
                return rows
            elif task == "Delete_logic_feature":
                cursor.execute(
                    "DELETE FROM Logic_Features WHERE Logic_Feature_Name =?", (value1,))
                connection.commit()
                return rows
            elif task == "Delete_range_feature":
                cursor.execute(
                    "DELETE FROM Range_Features WHERE Range_Feature_Name =?", (value1,))
                connection.commit()
                return rows
            elif task == "Delete_scalar_feature":
                cursor.execute(
                    "DELETE FROM Scalar_Features WHERE Scalar_Feature_Name =?", (value1,))
                connection.commit()
                return rows

        elif "check" in task:
            if task == "check_logic_value":
                cursor.execute(
                    "SELECT Logic_Value0, Logic_Value1 FROM Logic_Features WHERE Logic_Feature_Name =?.", (name))
                rows = cursor.fetchall()
                return rows
        elif "update" in task:
            if task == "update_Class_Range_Feature":

                cursor.execute("UPDATE Class_Range_Feature SET Value = ? WHERE Range_Feature = ? AND Class_Name= ?", (int(
                    value2), value1, name))
                connection.commit()
                rows = cursor.fetchall()
                return rows

            if task == "update_Class_Logic_Feature":

                cursor.execute(
                    "UPDATE Class_Logic_Features SET Value = ? WHERE Class_Name  = ? AND Logic_Feature = ?", (value2, name, value1,))
                connection.commit()
                rows = cursor.fetchall()
                return rows

            if task == "update_Class_Scalar_Feature":

                cursor.execute(
                    "UPDATE Class_Scalar_Feature SET Value = ? WHERE Class_Name  = ? AND Scalar_Feature = ?", (value2, name, value1,))
                connection.commit()
                rows = cursor.fetchall()
                return rows

            if task == "update_range_feature":

                cursor.execute("UPDATE Range_Features SET Range_Value_From = ?, Range_Value_To = ? WHERE Range_Feature_Name = ?", (int(
                    value1), int(value2), name,))
                connection.commit()
                rows = cursor.fetchall()
                return rows

        elif "clear" in task:

            if task == "clear_feature":

                feature_type = ""
                if name in [feature[0] for feature in self.SQL_Lite("get_logic_features")]:
                    feature_type = "logic"
                elif name in [feature[0] for feature in self.SQL_Lite("get_range_features")]:
                    feature_type = "range"
                elif name in [feature[0] for feature in self.SQL_Lite("get_scalar_features")]:
                    feature_type = "scalar"

                if feature_type == "scalar":

                    cursor.execute(
                        "DELETE FROM Scalar_Features WHERE Scalar_Feature_Name = ?;", (name,))
                    cursor.execute(
                        "INSERT INTO Scalar_Features (Scalar_Feature_Name) VALUES (?);", (name,))

                    connection.commit()
                    rows = cursor.fetchall()
                    return rows
                elif feature_type == "range":

                    cursor.execute(
                        "DELETE FROM Range_Features WHERE Range_Feature_Name = ?;", (name,))
                    cursor.execute(
                        "INSERT INTO Range_Features (Range_Feature_Name) VALUES (?);", (name,))

                    connection.commit()
                    rows = cursor.fetchall()
                    return rows

        elif "rem_Feature_from_" in task:

            if task == "rem_Feature_from_Class_Logic_Feature":
                cursor.execute(
                    "DELETE FROM Class_Logic_Features WHERE Class_Name is ? AND Logic_Feature IS ?", (name, value1))
                connection.commit()
                return rows

            if task == "rem_Feature_from_Class_Range_Feature":
                cursor.execute(
                    "DELETE FROM Class_Range_Feature WHERE Class_Name is ? AND Range_Feature IS ?", (name, value1))
                connection.commit()
                return rows

            if task == "rem_Feature_from_Class_Scalar_Feature":
                cursor.execute(
                    "DELETE FROM Scalar_Features WHERE Class_Name is ? AND Scalar_Feature IS ?", (name, value1))
                connection.commit()
                return rows

        connection.close()


app = App()
app.mainloop()

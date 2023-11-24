import pandas as pd


# Функция для расчета точности количества транспортных средств
def calculate_count_accuracy(predicted, ground_truth, threshold):
    return abs(predicted - ground_truth) <= ground_truth * threshold

# Функция для расчета точности средней скорости
def calculate_speed_accuracy(predicted, ground_truth, threshold):
    return abs(predicted - ground_truth) <= threshold

# Функция для расчета метрики на основе предсказаний и эталонных данных
def calculate_metric(ground_truth_csv, predictions_csv):
    # Загрузка данных
    ground_truth = pd.read_csv(ground_truth_csv)
    predictions = pd.read_csv(predictions_csv)
    
    # Инициализация счетчика для правильно предсказанных видео
    correct_predictions = 0
    
    # Проверка каждой записи
    for _, gt_row in ground_truth.iterrows():
        # Извлекаем соответствующую запись из предсказаний
        pred_row = predictions[predictions['file_name'] == gt_row['file_name']].iloc[0]
        
        # Проверяем точность для каждого класса транспортного средства
        car_count_correct = calculate_count_accuracy(pred_row['quantity_car'], gt_row['quantity_car'], 0.10)
        van_count_correct = calculate_count_accuracy(pred_row['quantity_van'], gt_row['quantity_van'], 0.20)
        bus_count_correct = calculate_count_accuracy(pred_row['quantity_bus'], gt_row['quantity_bus'], 0.20)
        
        car_speed_correct = calculate_speed_accuracy(pred_row['average_speed_car'], gt_row['average_speed_car'], 10)
        van_speed_correct = calculate_speed_accuracy(pred_row['average_speed_van'], gt_row['average_speed_van'], 10)
        bus_speed_correct = calculate_speed_accuracy(pred_row['average_speed_bus'], gt_row['average_speed_bus'], 10)
        
        # Если все проверки пройдены, увеличиваем счетчик
        if car_count_correct and van_count_correct and bus_count_correct and \
           car_speed_correct and van_speed_correct and bus_speed_correct:
            correct_predictions += 1
            
    # Рассчитываем итоговую метрику
    metric = correct_predictions / len(ground_truth)
    
    return metric

from src.preprocessing import create_generators

dataset_path = "data/raw/PlantVillage"

train_generator, validation_generator = create_generators(dataset_path)

print("Training Images:", train_generator.samples)
print("Validation Images:", validation_generator.samples)

print(train_generator.class_indices)
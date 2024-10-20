Here’s the **README** in plain text:

---

# Taxi-v3 Solver

This project solves the [Taxi-v3 environment](https://gymnasium.farama.org/environments/toy_text/taxi/) from Gymnasium using Q-learning. Taxi-v3 is a grid-based environment where the goal is to pick up passengers and drop them off at designated locations efficiently, minimizing steps and avoiding illegal actions.

## Environment Description

- **State space**: 500 discrete states (25 taxi positions, 5 passenger locations, and 4 destination locations)
- **Action space**: 6 discrete actions (move south, north, east, west, pickup, dropoff)
- **Goal**: Navigate the grid to complete tasks with minimal steps.

## How to Run

1. Clone this repo:
   ```
   git clone https://github.com/plagiagia/Reinforcement_Learning_Projects.git
   ```
2. Navigate to the project and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

## License

This project is licensed under the MIT License.

---

This format should work for you without breaking the output. Let me know if you need further adjustments!
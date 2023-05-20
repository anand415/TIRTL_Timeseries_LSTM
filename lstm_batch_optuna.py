import optuna
import joblib
study = optuna.create_study(study_name='batch_study',directions=["minimize"])
study.optimize(objective, n_trials=5)
joblib.dump(study, "batch_study.pkl")
for ii in range(0,30):
  study = joblib.load("study.pkl")
  # study = optuna.load_study(study_name="LSTMS")
  study.optimize(objective, n_trials=10)
  joblib.dump(study, "batch_study.pkl")
# study = joblib.load("study.pkl")
# fig = optuna.visualization.plot_contour(study)
# fig.show()

# fig = optuna.visualization.plot_intermediate_values(study)
# fig.show()

# fig = optuna.visualization.plot_optimization_history(study)
# fig.show()

study = joblib.load("batch_study.pkl")
anal_objective(study.best_trial)  # calculate acc, f1, recall, and precision
# new_model = tf.keras.models.load_model('saved_model/my_model')
[trainScore, valScore, testScore, trainPredict, valPredict, testPredict] = joblib.load("batch_studyanal.pkl")

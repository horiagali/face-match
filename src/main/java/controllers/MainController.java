package controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Label;

public class MainController {

    @FXML
    private Label helloLabel;

    @FXML
    public void initialize() {
        helloLabel.setText("Welcome to JavaFX!");
    }

    @FXML
    public void changeText() {
        helloLabel.setText("Button Clicked!");
    }
}

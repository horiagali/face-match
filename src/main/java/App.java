import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App {

    // Method to initialize the app, setting the primary stage and loading FXML
    public void initialize(Stage primaryStage) {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getClassLoader().getResource("fxml/MainScene.fxml"));
            Parent root = loader.load();

            Scene scene = new Scene(root);
            primaryStage.setScene(scene);

            // Set the window to fullscreen
            primaryStage.setFullScreen(true);

            primaryStage.setTitle("JavaFX Application");

            primaryStage.show();
        } catch (Exception e) {
            e.printStackTrace();  // Handle any exceptions that may occur while loading the FXML
        }
    }
}

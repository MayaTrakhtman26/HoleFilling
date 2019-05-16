import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_8UC1;

public class Main {

    static final private int NAME = 0;
    static final private int TYPE = 1;
    static final private String NAME_ADDITION = "_filled.";
    static final private String TYPE_SEPARATOR = "\\.";
    static final private float SCALING_FACTOR = 255;
    static final private int IMAGE = 0;
    static final private int MASK = 1;
    static final private int Z = 2;
    static final private int EPSILON = 3;
    static final private int CONNECTIVITY = 4;
    private static final int FOUR_CONNECTIVITY = 4;
    private static final int EIGHT_CONNECTIVITY = 8;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String imagePath = args[IMAGE];
        String maskPath = args[MASK];
        float z = Float.parseFloat(args[Z]);
        float epsilon = Integer.parseInt(args[EPSILON]);
        int connectivity = Integer.parseInt(args[CONNECTIVITY]);

        if (epsilon <= 0 || (connectivity != FOUR_CONNECTIVITY && connectivity != EIGHT_CONNECTIVITY)) {
            return;
        }

        Mat imageWithHole = mergeImageAndMask(imagePath, maskPath);

        HoleFilling holeFilling = new HoleFilling(imageWithHole, z, epsilon, connectivity);
        Mat filledImage = holeFilling.fill();

        filledImage.convertTo(filledImage, CV_8UC1, SCALING_FACTOR);
        String[] splitPath = imagePath.split(TYPE_SEPARATOR);
        Imgcodecs.imwrite(splitPath[NAME] + NAME_ADDITION + splitPath[TYPE], filledImage);
    }

    /**
     * Merges the given image and the given mask.
     * @param imagePath the original image path
     * @param maskPath the hole path
     * @return image containing a hole
     */
    private static Mat mergeImageAndMask(String imagePath, String maskPath) {
        float pixel;

        Mat image = readImage(imagePath);
        Mat mask = readImage(maskPath);
        Mat imageWithHole = image.clone();

        int rows = Math.min(image.rows(), mask.rows());
        int cols = Math.min(image.cols(), mask.cols());

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                pixel = (float) ((image.get(i, j)[0] + 1) * mask.get(i, j)[0] - 1);
                imageWithHole.put(i, j, pixel);
            }
        }

        return imageWithHole;
    }

    private static Mat readImage(String path){
        Mat image = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
        image.convertTo(image, CV_32FC1, 1.0 / SCALING_FACTOR);
        return image;
    }
}
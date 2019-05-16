import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.Vector;
import java.util.function.Function;

/**
 * A class that fills the hole in the image.
 */
public class HoleFilling {

    private static final int EIGHT_CONNECTIVITY = 8;
    private static final int POINT_U = 0;
    private static final int POINT_V = 1;
    private static final int PAIR_OF_POINTS = 2;

    private Mat image;
    private Function<Vector<Point>, Float> weightedFunction;
    private Vector<Point> hole;
    private Vector<Point> boundary;
    private int connectivity;

    /**
     * Constructor for the HoleFilling class.
     * @param image image that contains a hole
     * @param z parameter for the weight function
     * @param epsilon safety parameter for the weight function
     * @param connectivity required connectivity
     */
    public HoleFilling(Mat image, float z, float epsilon, int connectivity) {
        this.image = image;
        this.connectivity = connectivity;
        this.weightedFunction = createWeightedFunction(z, epsilon);
        this.hole = new Vector<Point>();
        this.boundary = new Vector<Point>();
    }

    /**
     * Fills the hole in the image based on the parameters received in the constructor.
     * @return the filled image
     */
    public Mat fill(){
        findHole();

        if(this.hole == null){
            return null;
        }

        findBoundary();
        return setValues();
    }

    /**
     * Sets the values of the hole based on the hole and boundary previously found.
     * @return the filled image
     */
    private Mat setValues(){
        Mat filledImage = image.clone();
        Vector<Point> points = new Vector<Point>(PAIR_OF_POINTS);
        float numerator, denominator, weight;

        for(Point u: this.hole){
            numerator = 0;
            denominator = 0;
            points.add(u);

            for(Point v: this.boundary){
                points.add(v);
                weight = this.weightedFunction.apply(points);
                numerator += weight*this.image.get((int) v.x, (int) v.y)[0];
                denominator += weight;
                points.remove(POINT_V);
            }

            points.clear();
            filledImage.put((int) u.x,(int) u.y, numerator/denominator); // sets a value in the hole
        }
        return filledImage;
    }

    /**
     * Finds the hole in the image.
     */
    private void findHole(){
        int rows = this.image.rows();
        int cols = this.image.cols();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                if(this.image.get(i, j)[0] < 0){ // if the pixel is in the hole
                    this.hole.add(new Point(i,j));
                }
            }
        }
    }

    /**
     * Finds the boundary.
     */
    private void findBoundary(){
        Vector<Point> pointNeighbourhood;

        for(Point point : this.hole){
            pointNeighbourhood = getNeighbourhood(point);
            for(Point neighbour : pointNeighbourhood){
                if(this.image.get((int) neighbour.x, (int) neighbour.y)[0] >= 0){ // if a neighbour is not in the hole
                    this.boundary.add(neighbour);
                }
            }
        }
    }

    /**
     * Finds the neighbours of a point based on the connectivity.
     * @param point for which to find the neighbours
     * @return neighbours of the function
     */
    private Vector<Point> getNeighbourhood(Point point){
        Vector<Point> pointNeighbourhood = new Vector<Point>();

        int i = (int) point.x;
        int j = (int) point.y;

        pointNeighbourhood.add(new Point(i+1, j));
        pointNeighbourhood.add(new Point(i-1, j));
        pointNeighbourhood.add(new Point(i, j+1));
        pointNeighbourhood.add(new Point(i, j-1));

        if(this.connectivity == EIGHT_CONNECTIVITY){
            return getNeighbourhood8connectivity(pointNeighbourhood, point);
        }

        return pointNeighbourhood;
    }

    /**
     * Adds more neighbours to receive 8 connectivity.
     * @param pointNeighbourhood 4 connectivity neighbours
     * @param point for which to find the neighbours
     * @return neighbours of the function
     */
    private Vector<Point> getNeighbourhood8connectivity(Vector<Point> pointNeighbourhood, Point point){

        int i = (int) point.x;
        int j = (int) point.y;

        pointNeighbourhood.add(new Point(i+1, j-1));
        pointNeighbourhood.add(new Point(i-1, j-1));
        pointNeighbourhood.add(new Point(i+1, j+1));
        pointNeighbourhood.add(new Point(i-1, j+1));

        return pointNeighbourhood;
    }

    /**
     * Returns the weighted function based on the given parameters.
     * @param z parameter for the weight function
     * @param epsilon safety parameter for the weight function
     * @return weighted function
     */
    private Function<Vector<Point>, Float> createWeightedFunction(float z, float epsilon){
        return (Vector<Point> points) -> {
            double xDistance = points.get(POINT_U).x - points.get(POINT_V).x;
            double yDistance = points.get(POINT_U).y - points.get(POINT_V).y;
            double euclideanDistanceSquare = (Math.pow(xDistance, 2) + Math.pow(yDistance, 2));
            return (float) 1.0 / ((float) Math.pow(euclideanDistanceSquare, z/2.0) + epsilon);};
    }
}


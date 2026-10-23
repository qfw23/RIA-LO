// RIA-LO LiDAR acquisition parameters.

#ifndef RIA_LO_LIDAR_H_
#define RIA_LO_LIDAR_H_

namespace lidar
{
class Lidar
{
public:
    Lidar();

    void setScanPeriod(double scan_period_in);
    void setLines(double num_lines_in);
    void setMaxDistance(double max_distance_in);
    void setMinDistance(double min_distance_in);

    double max_distance;
    double min_distance;
    int num_lines;
    double scan_period;
};
}  // namespace lidar

#endif  // RIA_LO_LIDAR_H_


#include <3DHoPD/3DHoPD.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <memory>

// Silence all warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wattributes"
#define VTK_LEGACY_SILENT
#define PCL_NO_PRECOMPILE

float calculateCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    float resolution = 0.0f;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); i += 10)
    { // Sample every 10th point
        std::vector<int> indices(2);
        std::vector<float> distances(2);
        if (kdtree.nearestKSearch((*cloud)[i], 2, indices, distances) == 2)
        {
            resolution += sqrt(distances[1]);
        }
    }
    return resolution / (cloud->size() / 10);
}

void alignPrincipalAxes(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr target)
{
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(source);
    Eigen::Matrix3f src_axes = pca.getEigenVectors();
    if (src_axes.determinant() < 0)
        src_axes.col(2) *= -1;

    pca.setInputCloud(target);
    Eigen::Matrix3f tgt_axes = pca.getEigenVectors();
    if (tgt_axes.determinant() < 0)
        tgt_axes.col(2) *= -1;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = tgt_axes * src_axes.transpose();
    transform.block<3, 1>(0, 3) = pca.getMean().head<3>() - transform.block<3, 3>(0, 0) * pca.getMean().head<3>();

    pcl::transformPointCloud(*source, *source, transform);
}

int main()
{

    vtkObject::GlobalWarningDisplayOff();

    // =============== SET PATHS ===============
    std::string object_path = "/home/rohan-vinkare/Collage/BTP/3DHOPD-live/data_collected/f2.pcd";
    std::string scene_path = "/home/rohan-vinkare/Collage/BTP/3DHOPD-live/data_collected/g7.pcd";
    // =========================================

    // Load point clouds
    auto object = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto scene = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(object_path, *object) == -1 ||
        pcl::io::loadPCDFile<pcl::PointXYZ>(scene_path, *scene) == -1)
    {
        std::cerr << "Error loading point clouds!" << std::endl;
        return -1;
    }

    // Calculate resolutions
    float object_res = calculateCloudResolution(object);
    float scene_res = calculateCloudResolution(scene);
    float base_res = std::max(object_res, scene_res);

    std::cout << "Resolutions - Object: " << object_res
              << " Scene: " << scene_res << std::endl;

    // Preprocessing
    auto start = std::chrono::high_resolution_clock::now();

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    float leaf_size = base_res * 3.0f;
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);

    auto object_processed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto scene_processed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    voxel_filter.setInputCloud(object);
    voxel_filter.filter(*object_processed);
    voxel_filter.setInputCloud(scene);
    voxel_filter.filter(*scene_processed);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setMeanK(30);
    sor.setStddevMulThresh(1.5f);
    sor.setInputCloud(object_processed);
    sor.filter(*object_processed);
    sor.setInputCloud(scene_processed);
    sor.filter(*scene_processed);

    alignPrincipalAxes(object_processed, scene_processed);

    // 3D HoPD processing
    threeDHoPD object_descriptor, scene_descriptor;
    object_descriptor.cloud = *object_processed;
    scene_descriptor.cloud = *scene_processed;

    float object_kp_res = leaf_size * 1.8f;
    float scene_kp_res = leaf_size * 2.2f;

    object_descriptor.detect_uniform_keypoints_on_cloud(object_kp_res);
    scene_descriptor.detect_uniform_keypoints_on_cloud(scene_kp_res);

    std::cout << "Keypoints - Object: " << object_descriptor.cloud_keypoints.size()
              << " Scene: " << scene_descriptor.cloud_keypoints.size() << std::endl;

    object_descriptor.kdtree.setInputCloud(object_processed);
    scene_descriptor.kdtree.setInputCloud(scene_processed);

    float object_radius = leaf_size * 8.0f;
    float scene_radius = leaf_size * 6.0f;

    object_descriptor.JUST_REFERENCE_FRAME_descriptors(object_radius);
    scene_descriptor.JUST_REFERENCE_FRAME_descriptors(scene_radius);

    // Matching
    pcl::Correspondences corrs;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_LRF;
    pcl::PointCloud<pcl::PointXYZ> lrf_cloud;

    for (const auto &desc : scene_descriptor.cloud_LRF_descriptors)
    {
        pcl::PointXYZ point;
        point.x = desc.vector[0];
        point.y = desc.vector[1];
        point.z = desc.vector[2];
        lrf_cloud.push_back(point);
    }
    kdtree_LRF.setInputCloud(lrf_cloud.makeShared());

    float match_threshold = leaf_size * 3.0f;
    float max_descriptor_dist = 0.3f;

    for (size_t i = 0; i < object_descriptor.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ search_point;
        search_point.x = object_descriptor.cloud_LRF_descriptors[i].vector[0];
        search_point.y = object_descriptor.cloud_LRF_descriptors[i].vector[1];
        search_point.z = object_descriptor.cloud_LRF_descriptors[i].vector[2];

        std::vector<int> nn_indices;
        std::vector<float> nn_dists;

        if (kdtree_LRF.radiusSearch(search_point, match_threshold, nn_indices, nn_dists) > 0)
        {
            std::vector<std::pair<float, int>> candidates;

            for (size_t j = 0; j < nn_indices.size(); j++)
            {
                Eigen::VectorXf obj_vec(15), scene_vec(15);
                for (int k = 0; k < 15; k++)
                {
                    obj_vec[k] = object_descriptor.cloud_distance_histogram_descriptors[i].vector[k];
                    scene_vec[k] = scene_descriptor.cloud_distance_histogram_descriptors[nn_indices[j]].vector[k];
                }
                float dist = (obj_vec - scene_vec).norm();
                if (dist < max_descriptor_dist)
                {
                    candidates.emplace_back(dist, nn_indices[j]);
                }
            }

            if (!candidates.empty())
            {
                std::sort(candidates.begin(), candidates.end());
                pcl::Correspondence corr;
                corr.index_query = static_cast<int>(i);
                corr.index_match = candidates[0].second;
                corr.distance = candidates[0].first;
                corrs.push_back(corr);
            }
        }
    }

    std::cout << "Initial matches: " << corrs.size() << std::endl;

    // RANSAC
    pcl::CorrespondencesConstPtr corrs_ptr = std::make_shared<pcl::Correspondences>(corrs);
    pcl::Correspondences inliers;

    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> ransac;
    ransac.setInputSource(object_descriptor.cloud_keypoints.makeShared());
    ransac.setInputTarget(scene_descriptor.cloud_keypoints.makeShared());
    ransac.setInlierThreshold(leaf_size * 4.0f);
    ransac.setMaximumIterations(50000);
    ransac.setInputCorrespondences(corrs_ptr);
    ransac.getCorrespondences(inliers);

    Eigen::Matrix4f transformation = ransac.getBestTransformation();
    std::cout << "RANSAC inliers: " << inliers.size() << std::endl;

    // ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaximumIterations(100);
    icp.setMaxCorrespondenceDistance(leaf_size * 5.0f);
    icp.setTransformationEpsilon(1e-8);

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_object(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*object_processed, *transformed_object, transformation);

    icp.setInputSource(transformed_object);
    icp.setInputTarget(scene_processed);
    pcl::PointCloud<pcl::PointXYZ> final_alignment;
    icp.align(final_alignment);

    if (icp.hasConverged())
    {
        transformation = icp.getFinalTransformation() * transformation;
        std::cout << "ICP refinement converged. Fitness: " << icp.getFitnessScore() << std::endl;
    }

    // Evaluation
    int correct_matches = 0;
    float eval_threshold = leaf_size * 5.0f;
    pcl::Correspondences strong_inliers;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_eval;
    kdtree_eval.setInputCloud(scene_descriptor.cloud_keypoints.makeShared());

    for (const auto &inlier : inliers)
    {
        Eigen::Vector4f transformed = transformation *
                                      object_descriptor.cloud_keypoints[inlier.index_query].getVector4fMap();
        pcl::PointXYZ search_pt;
        search_pt.getVector4fMap() = transformed;

        std::vector<int> nn_idx(1);
        std::vector<float> nn_dist(1);
        if (kdtree_eval.nearestKSearch(search_pt, 1, nn_idx, nn_dist) > 0)
        {
            if (std::sqrt(nn_dist[0]) < eval_threshold)
            {
                correct_matches++;
                strong_inliers.push_back(inlier);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n============== Results ==============" << std::endl;
    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
    std::cout << "Correct matches: " << correct_matches << "/" << inliers.size() << std::endl;
    std::cout << "Precision: " << static_cast<float>(correct_matches) / inliers.size() << std::endl;
    std::cout << "Transformation Matrix:\n"
              << transformation << std::endl;

    // FIXED VISUALIZATION
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D HoPD Results"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setCameraPosition(0, 0, -3, 0, -1, 0);

    // Add coordinate system first
    viewer->addCoordinateSystem(0.5);

    // Visualize scene (green)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_handler(scene_processed, 0, 255, 0);
    viewer->addPointCloud(scene_processed, scene_handler, "scene");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");

    // Visualize aligned object (red)
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_object(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*object_processed, *aligned_object, transformation);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> object_handler(aligned_object, 255, 0, 0);
    viewer->addPointCloud(aligned_object, object_handler, "object");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object");

    // Add correspondences with proper indices
    if (!strong_inliers.empty())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_kp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_kp(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto &corr : strong_inliers)
        {
            object_kp->push_back(object_descriptor.cloud_keypoints[corr.index_query]);
            scene_kp->push_back(scene_descriptor.cloud_keypoints[corr.index_match]);
        }

        viewer->addPointCloud(object_kp, "object_kp");
        viewer->addPointCloud(scene_kp, "scene_kp");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "object_kp");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_kp");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 0, "object_kp");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 0, "scene_kp");

        // Add lines between correspondences
        for (size_t i = 0; i < strong_inliers.size(); ++i)
        {
            std::string line_id = "line_" + std::to_string(i);
            pcl::PointXYZ pt1 = object_kp->points[i];
            pcl::PointXYZ pt2 = scene_kp->points[i];
            viewer->addLine<pcl::PointXYZ>(pt1, pt2, 1, 1, 0, line_id);
        }
    }

    // Add text labels
    viewer->addText("Scene (Green)", 10, 20, 14, 0, 1, 0, "scene_text");
    viewer->addText("Aligned Object (Red)", 10, 40, 14, 1, 0, 0, "object_text");
    if (!strong_inliers.empty())
    {
        viewer->addText("Matches: " + std::to_string(strong_inliers.size()), 10, 60, 14, 1, 1, 0, "matches_text");
    }

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}

#pragma GCC diagnostic pop

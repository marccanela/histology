
    #===
        plt.imshow(binary_image);
        masked_blurred = blurred_normalized.copy()
        masked_blurred[binary_image] = 0    
        plt.imshow(masked_blurred);
        
        # from skimage import filters
        # denoised = ndi.median_filter(masked_blurred, size=3)
        # li_thresholded = denoised > filters.threshold_li(denoised)
        # plt.imshow(li_thresholded);
        
        from skimage import morphology
        width = math.sqrt(min_area_threshold_pixels)
        remove_holes = morphology.remove_small_holes(li_thresholded, width ** 3)
        remove_objects = morphology.remove_small_objects(remove_holes, width ** 3)
        plt.imshow(remove_objects);
        
        labeled_array, num_clusters = label(remove_objects)
        regions = regionprops(labeled_array)
        areas = [region.area for region in regions]
        median_area = np.median(areas)
        median_diameter = int(math.sqrt((4 * median_area) / math.pi))
        
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        distance = ndi.distance_transform_edt(remove_objects)        
        plt.imshow(distance);
        
        
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=remove_objects, min_distance=int(median_diameter/1.5))
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        plt.imshow(markers);
        labels = watershed(-distance, markers, mask=remove_objects)
        
        from matplotlib.colors import LinearSegmentedColormap
        def create_custom_cmap(data):
            n = np.max(data)
            base_colors = plt.cm.viridis(np.linspace(0, 1, n))
            np.random.shuffle(base_colors)
            cmap = LinearSegmentedColormap.from_list('custom_cmap', base_colors)
            return cmap
        custom_cmap = create_custom_cmap(labels)
        plt.imshow(labels, cmap=custom_cmap);

        
    #===
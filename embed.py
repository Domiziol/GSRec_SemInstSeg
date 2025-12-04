#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:47:32 2025

@author: artur
"""
import torch
import torch.nn as nn

# Select device
torch.set_default_device('cpu')  
#torch.set_default_device('cuda')  

# Data
# Points visible in views: 2,3,5,7,11,8,14,15,17,21
# Anticipated real-world segments {2,7,11}, {3,5,15,20}, {8,14,17,21}
views_segments = []
view_segment = torch.zeros(30, dtype=torch.long) # zero - means point not present in a view
view_segment[[2,7,11]] = 1 # points 2,7,11 in the same segment (#1 in view 1)
view_segment[[3,5,15]] = 2 # points 3,5,15 in the same segment (#2 in view 1)
views_segments.append(view_segment) 
view_segment = torch.zeros(30, dtype=torch.long) # zero - means point not present in a view
view_segment[[8,14,17,21]] = 3 # points 8,14,17,21 in the same segment (#3 in view 2)
view_segment[[15,20]] = 4 # note: points 15,20 in the same segment (#4 in view 2)
views_segments.append(view_segment) 
# concern: above the global segments id is provided, whereas we don't have that information. Instead there will be a local id

# Variables and optimizer
point_embeddings = nn.Embedding(30, 3)
optimizer = torch.optim.Adam(point_embeddings.parameters(), lr=0.1)

# Computation of constants
views_pairs, views_point_matches = [], []
for view_segments in views_segments:
    view_points = torch.squeeze(torch.nonzero(view_segments))
    view_pairs = torch.combinations(view_points) # pair of all points that are in some mask (nonzero)
    view_point_segments = view_segments[view_pairs] # pair af local mask id for each pair 
    view_point_matches = (view_point_segments[:,0]==view_point_segments[:,1]).float()*2-1 # for a pair of local masks id, add 1 if both values are the same and -1 if not
    views_pairs.append(view_pairs) # global for points pairs ... this might be a lot of points!
    views_point_matches.append(view_point_matches) # 1 or -1 per pair of points - global
    
# Optimization
for step in range(200):
    losses = []
    for view_pairs, view_point_matches in zip(views_pairs, views_point_matches):
        view_embedding_pairs = nn.functional.normalize(point_embeddings(view_pairs), dim=2) # and this is what?
        view_embedding_sqdists = (view_embedding_pairs[:,0,:] - view_embedding_pairs[:,1,:]).square().sum(dim=1) # what are those indexes :,0,:?
        loss = torch.dot(view_point_matches, view_embedding_sqdists)
        losses.append(loss)        
        #print(torch.cat((view_pairs, torch.unsqueeze(view_point_matches,1)),1))
    
    # Optimization step
    total_loss = torch.stack(losses).sum()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(total_loss)

# Results    
embds = nn.functional.normalize(point_embeddings.weight, dim=1).detach().cpu().numpy()
print(embds[[2,7,11],:])
print("-------")
print(embds[[3,5,15,20],:])
print("-------")
print(embds[[8,14,17,21],:])


# GitHub Repository Setup Guide

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Repository name: `er_triage_workflow` (or your preferred name)
4. Description: "LangGraph-based workflow for ER triage admission decisions"
5. Choose visibility (Public or Private)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 2: Add Remote and Push

After creating the repository, GitHub will show you the repository URL. Use one of these commands:

### If using HTTPS:
```bash
cd "/Users/yuhsuanko/Desktop/UChicago/UChicago_Q4/Capstone II/Capstone_Organized"
git remote add origin https://github.com/YOUR_USERNAME/er_triage_workflow.git
git branch -M main
git push -u origin main
```

### If using SSH:
```bash
cd "/Users/yuhsuanko/Desktop/UChicago/UChicago_Q4/Capstone II/Capstone_Organized"
git remote add origin git@github.com:YOUR_USERNAME/er_triage_workflow.git
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME` with your GitHub username!**

## Step 3: Verify

After pushing, verify by visiting your repository on GitHub. You should see all the files we created.

## Troubleshooting

### If you get authentication errors:
- For HTTPS: You may need to use a Personal Access Token instead of password
- For SSH: Make sure your SSH key is added to GitHub

### If you want to push only the er_triage_workflow folder:
If you want the repository to contain only the `er_triage_workflow` folder (not the entire parent directory), you can:

1. Create a new repository
2. Initialize git inside `er_triage_workflow`:
   ```bash
   cd "/Users/yuhsuanko/Desktop/UChicago/UChicago_Q4/Capstone II/Capstone_Organized/er_triage_workflow"
   git init
   git add .
   git commit -m "Initial commit: ER Triage Workflow package"
   git remote add origin https://github.com/YOUR_USERNAME/er_triage_workflow.git
   git push -u origin main
   ```


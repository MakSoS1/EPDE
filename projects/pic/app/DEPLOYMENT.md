# Streamlit Community Cloud

Deploy the existing application from the GitHub fork. No separate website or
interface is needed. Community Cloud requires the app and its dependencies to
have been pushed to the selected GitHub branch; it cannot deploy local changes.

Use these settings at https://share.streamlit.io:

| Setting | Value |
| --- | --- |
| Repository | `MakSoS1/EPDE` |
| Branch | The branch containing the reviewed app changes |
| Main file path | `projects/pic/app/Home.py` |
| Python version (Advanced settings) | `3.13` |

No application secrets are required. `requirements.txt` beside `Home.py` installs
the app dependencies and Linux CPU-only PyTorch. The app and its subprocesses
import EPDE from this repository, preserving the version under review. Notebook
execution dependencies are not needed for the web application.

After the build, verify the application at its public URL:

1. Open Overview, Data sets and Derivatives; check that plots render.
2. On How EPDE works, run the short evolution and inspect the equation and fronts.
3. Run one default oscillator search with one worker. Check its terminal status
   and open the saved result. This tests background processes as well as the UI.
4. Test a small upload and open its saved result.

Community Cloud has limited CPU and memory. Successful local checks do not prove
that a full benchmark fits those limits; validate small runs on the hosted app
before inviting reviewers. Hosted searches run on Community Cloud, not on the
visitor's computer.

Results, uploads and job controls currently belong to the shared application
instance; they are not isolated per viewer. Use public demonstration data. Treat
cloud-generated files as temporary and download results you need to keep before
restarting or deleting the app.

To end the review demo, delete the app from its Community Cloud management menu.
This does not delete the GitHub repository.

Official documentation:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

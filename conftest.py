import pytest

def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--plots", action="store_true", default=False, help="Enable plotting for debugging"
    )

@pytest.fixture
def test_plots(request):
    """Fixture to get the plots flag."""
    return request.config.getoption("--plots")

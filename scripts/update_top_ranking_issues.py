import os
from datetime import datetime, timedelta

from github import Github
from github.Issue import Issue
from github.Repository import Repository


def main():
    start_time: datetime = datetime.now()

    # --- Initialization ---
    # GitHub Workflow will pass in the token as an environment variable,
    # but we can place it in our env when running the script locally,
    # for convenience
    local_github_token: str | None = None
    github_token: str | None = (
        local_github_token or os.getenv("GITHUB_ACCESS_TOKEN")
    )
    github = Github(github_token)

    # repository name
    repo_name: str = "pvlib/pvlib-python"
    repository: Repository = github.get_repo(repo_name)

    # Rate limiting
    remaining_requests_before: int = github.rate_limiting[0]
    print(f"Remaining requests before: {remaining_requests_before}")

    # --- Actions ---
    # Get sorted issues
    query: str = (
        f'repo:{repository.full_name} is:open is:issue sort:reactions-+1-desc'
    )
    issues = github.search_issues(query)

    # Format markdown
    ranked_issues = []
    for i, issue in enumerate(issues):
        markdown_bullet_point: str = (
            f"{issue.html_url} " +
            f"({issue._rawData["reactions"]["+1"]} :thumbsup:)"
        )

        markdown_bullet_point = f"{i + 1}. {markdown_bullet_point}"
        ranked_issues.append(markdown_bullet_point)

        if i >= 19:
            break

    # edit top issues
    top_issues_card: Issue = repository.get_issue(number=2196)
    header = "Top Ranking Issues"
    new_body = header + "\n" + "\n".join(ranked_issues)
    top_issues_card.edit(
        body=new_body
    )

    print(top_issues_card.body)

    run_duration: timedelta = datetime.now() - start_time
    print(run_duration)


if __name__ == "__main__":
    main()
